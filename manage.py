from argcmdr import Local, LocalRoot, localmethod


class Aequitas(LocalRoot):
    """manage the aequitas project"""


@Aequitas.register
class Web(Local):
    """manage the aequitas webapp"""

    class Dev(Local):
        """run the web development server"""

        def prepare(self):
            return self.local['python']['-m', 'aequitas_webapp.basic_upload']

    class Env(Local):
        """manage the webapp environment"""

        ENV = 'aequitas-pro'

        @localmethod('-n', '--name', default=ENV,
                    help=f"environment console to open (default: {ENV})")
        def console(self, args):
            """open the environment web console"""
            return self.local['eb']['console', args.name]

        class Create(Local):
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
                command = self.local['eb'][
                    'create',
                    '-nh', # return immediately
                    '-s',  # stand-alone for now (no ELB)
                    args.name,
                ]

                if args.version:
                    command = command['--version', args.version]

                yield command

        class Deploy(Local):
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
                return self.local['eb'][
                    'deploy',
                    '-nh',
                    '-l', args.version,
                    args.name,
                ]
