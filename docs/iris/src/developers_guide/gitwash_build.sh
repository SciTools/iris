#!/usr/bin/env sh

# Get the latest gitwash_dumper.py from GitHub.
echo "Downloading latest gitwash_dumper.py from GitHub ..."
echo
curl -O https://raw.github.com/matthew-brett/gitwash/master/gitwash_dumper.py

# Generate the gitwash sub-directory.
echo
echo "Building gitwash ..."
echo
python gitwash_dumper.py --repo-name=iris --github-user=SciTools --gitwash-url=https://github.com/matthew-brett/gitwash.git --project-url=http://scitools.org.uk/ --project-ml-url=http://scitools.org.uk/mailman/listinfo ./ iris
