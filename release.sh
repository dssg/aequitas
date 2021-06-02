#!/bin/bash
# Shell script to deploy Aequitas
# 1st argument is version to bump ('major', 'minor', 'patch')
# 2nd argument is to pass to build ('-l')
# 3rd argument is repository to send ('testpypi', 'pypi')

helpFunction()
{
   echo ""
   echo "Usage: $0 -part ('major', 'minor', 'patch') -l -r ('testpypi', 'pypi')"
   echo -e "\t-p Version part to be bumped."
   echo -e "\t-l Lite build."
   echo -e "\t-r Release repository."
   exit 1 # Exit script after printing help
}
while getopts "p:r:l:" opt
do
   case "$opt" in
      p ) parameterP="$OPTARG" ;;
      l ) parameterL="-l" ;;
      r ) parameterR="-r $OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done



if [ -z "$parameterP" ]
then
   echo "Version part to bump must be explicit.";
   helpFunction
fi

if [ -z "$parameterL" ]
then
  parameterL=""
fi

if [ -z "$parameterR" ]
then
  parameterR=""
fi

# Begin script in case all parameters are correct

echo "$parameterP"
echo "$parameterL"
echo "$parameterR"

manage release bump $parameterP

manage release build $parameterL

manage release upload $parameterR
