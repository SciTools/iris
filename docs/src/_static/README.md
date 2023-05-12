# Iris logos

[![iris-logo-title.svg](iris-logo-title.svg)](iris-logo-title.svg)

Code for generating the logos is at:
[SciTools/marketing/iris/logo/generate_logo.py](https://github.com/SciTools/marketing/blob/master/iris/logo/generate_logo.py)

See the docstring of the `generate_logo()` function for more information.

## Why a scripted logo?

SVG logos are ideal for source-controlled projects:

* Low file size, with infinitely scaling quality
* Universally recognised vector format, editable by many software packages
* XML-style content = human-readable diff when changes are made

But Iris' logo is difficult to reproduce/edit using an SVG editor alone:

* Includes correctly projected, low resolution coastlines
* Needs precise alignment of the 'visual centre' of the iris with the centres
  of the Earth and the image

An SVG image is simply XML format, so can be easily assembled automatically
with a script, which can also be engineered to address the above problems.

Further advantages of using a script:

* Parameterised text, making it easy to standardise the logo across all Iris
  packages
* Can generate an animated GIF/SVG of a rotating Earth
