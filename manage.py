from pathlib import Path
from argparse import REMAINDER

from argcmdr import CacheDict, Local, LocalRoot, localmethod
from plumbum import local



ROOT_PATH = Path(__file__).parent

VENV = '.manage'
VENV_BIN_PATH = ROOT_PATH / VENV / 'bin'


def get_project_local(exe):
    (head, path) = local.env['PATH'].split(':', 1)
    assert VENV_BIN_PATH.samefile(head)
    with local.env(PATH=path):
        return local[exe]


project_local = CacheDict(get_project_local)


class Aequitas(LocalRoot):
    """manage the aequitas project"""


@Aequitas.register
class Container(Local):
    """manage the aequitas docker image and container"""

    image_name = 'aequitas'
    container_name = 'aequitas-dev'

    @localmethod
    def build(self):
        """build image"""
        return self.local['docker'][
            'build',
            '-t',
            self.image_name,
            ROOT_PATH,
        ]

    @localmethod
    def create(self):
        """create local container"""
        return self.local['docker'][
            'create',
            '-p', '5000:5000',
            '-e', 'HOST=0.0.0.0',
            '--name', self.container_name,
            self.image_name,
        ]

    @localmethod
    def start(self):
        """start local container"""
        return self.local['docker'][
            'start',
            self.container_name,
        ]

    @localmethod
    def stop(self):
        """stop local container"""
        return self.local['docker'][
            'stop',
            self.container_name,
        ]


@Aequitas.register
class Web(Local):
    """manage the aequitas webapp"""

    class Dev(Local):
        """run the web development server"""

        def prepare(self):
            return (
                self.local.FG,
                project_local['python']['-m', 'serve']
            )

    class Env(Local):
        """manage the webapp environment"""

        ENV = 'aequitas-pro'

        def __init__(self, parser):
            parser.add_argument(
                '-n', '--name',
                default=self.ENV,
                help=f"environment name (default: {self.ENV})",
            )

        @localmethod
        def console(self, args):
            """open the environment web console"""
            return (self.local.FG, self.local['eb']['console', args.name])

        @localmethod
        def logs(self, args):
            """read environment logs"""
            return (self.local.FG, self.local['eb']['logs', args.name])

        @localmethod
        def ssh(self, args):
            """ssh into the EC2 instance"""
            return (self.local.FG, self.local['eb']['ssh', args.name])

        class Create(Local):
            """create an environment"""

            def __init__(self, parser):
                parser.add_argument(
                    '-v', '--version',
                    help='previous version label to deploy to new environment',
                )

            def prepare(self, args):
                # AWS_EB_PROFILE=dsapp-ddj
                command = self.local['eb'][
                    'create',
                    '-nh',  # return immediately
                    '-s',   # stand-alone for now (no ELB)
                    args.name,
                ]

                if args.version:
                    command = command['--version', args.version]

                yield command

        class Deploy(Local):
            """deploy to an environment"""

            def __init__(self, parser):
                parser.add_argument(
                    'version',
                    help='version label to apply',
                )

            def prepare(self, args):
                return (
                    self.local.FG,
                    self.local['eb'][
                        'deploy',
                        '-nh',
                        '-l', args.version,
                        args.name,
                    ]
                )



@Aequitas.register
class Release(Local):
    """manage the aequitas releases and upload to pypi"""

    package_name = 'aequitas'

    bump_default_message = "Bump version: {current_version} → {new_version}"

    @localmethod('part', choices=('major', 'minor', 'patch'),
                 help="part of the version to be bumped")
    @localmethod('-m', '--message',
                 help=f"Tag message (in addition to default: "
                      f"'{bump_default_message}')")
    def bump(self, args):
        """increment package version"""
        if args.message:
            message = f"{self.bump_default_message}\n\n{args.message}"
        else:
            message = self.bump_default_message

        return self.local['bumpversion'][
            '--message', message,
            args.part,
        ]

    @local
    def build(self):
        """build the python distribution"""
        return (self.local.FG, self.local['python'][
            'setup.py',
            'sdist',
            'bdist_wheel',
        ])

    @localmethod('versions', metavar='version', nargs='*',
                 help="specific version(s) to upload (default: all)")
    def upload(self, args):
        """upload distribution(s) to pypi"""
        if args.versions:
            targets = [f'dist/{self.package_name}-{version}*'
                       for version in args.versions]
        else:
            targets = [f'dist/{self.package_name}-*']
        return (self.local.FG, self.local['twine']['upload'][targets])
