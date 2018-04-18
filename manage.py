import functools

from argcmdr import cmd, Local, LocalRoot


class BeanstalkCommand(Local):

    @property
    def eb(self):
        return self.local['.manage/bin/eb']


ebmethod = functools.partial(cmd, base=BeanstalkCommand)


class Aequitas(LocalRoot):
    """manage the aequitas project"""


@Aequitas.register
class Web(Local):
    """manage the aequitas webapp"""

    class Dev(Local):
        """run the web development server"""

        def prepare(self):
            return (
                self.local.FG,
                self.local['python']['-m', 'aequitas_webapp.basic_upload']
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

        @ebmethod
        def console(self, args):
            """open the environment web console"""
            return (self.local.FG, self.eb['console', args.name])

        @ebmethod
        def logs(self, args):
            """read environment logs"""
            return (self.local.FG, self.eb['logs', args.name])

        @ebmethod
        def ssh(self, args):
            """ssh into the EC2 instance"""
            return (self.local.FG, self.eb['ssh', args.name])

        class Create(BeanstalkCommand):
            """create an environment"""

            def __init__(self, parser):
                parser.add_argument(
                    '-v', '--version',
                    help='previous version label to deploy to new environment',
                )

            def prepare(self, args):
                # AWS_EB_PROFILE=dsapp-ddj
                command = self.eb[
                    'create',
                    '-nh', # return immediately
                    '-s',  # stand-alone for now (no ELB)
                    args.name,
                ]

                if args.version:
                    command = command['--version', args.version]

                yield command

        class Deploy(BeanstalkCommand):
            """deploy to an environment"""

            def __init__(self, parser):
                parser.add_argument(
                    'version',
                    help='version label to apply',
                )

            def prepare(self, args):
                return self.eb[
                    'deploy',
                    '-nh',
                    '-l', args.version,
                    args.name,
                ]
