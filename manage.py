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

        @ebmethod('-n', '--name', default=ENV,
                  help=f"environment console to open (default: {ENV})")
        def console(self, args):
            """open the environment web console"""
            return self.eb['console', args.name]

        class Create(BeanstalkCommand):
            """create an environment"""

            def __init__(self, parser):
                parser.add_argument(
                    '-v', '--version',
                    help='previous version label to deploy to new environment',
                )
                parser.add_argument(
                    '-n', '--name',
                    default=Web.Env.ENV,
                    help=f"name the environment (default: {Web.Env.ENV})",
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
                    '-n', '--name',
                    default=Web.Env.ENV,
                    help=f"environment to which to deploy (default: {Web.Env.ENV})",
                )
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
