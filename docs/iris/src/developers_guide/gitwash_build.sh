#!/usr/bin/env sh


# Generate the gitwash sub-directory.
echo
echo "Building gitwash ..."
echo
python gitwash_dumper.py --repo-name=iris --github-user=SciTools --gitwash-url=https://github.com/matthew-brett/gitwash.git --project-url=http://scitools.org.uk/iris --project-ml-url=http://scitools.org.uk/mailman/listinfo ./ iris
