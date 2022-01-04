# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

import json
import re
from typing import Dict, List

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective

ImageRepo = Dict[str, List[str]]

HASH_MATCH = re.compile(r"([^\/]+)\.png$")


def hash_from_url(url: str) -> str:
    match = HASH_MATCH.search(url)
    if not match:
        raise ValueError(f"url {url} does not match form `http...hash.png`")
    else:
        return match.groups()[0]


class ImageTestDirective(SphinxDirective):
    def run(self):
        with open(self.config["image_test_json"], "r") as fh:
            imagerepo = json.load(fh)
        enum_list = nodes.enumerated_list()
        nodelist = []
        nodelist.append(enum_list)
        for test in sorted(imagerepo):
            link_node = nodes.raw(
                "",
                f'<a href="{self.config["html_baseurl"]}/generated/image_test/{test}.html" />{test}</a>',
                format="html",
            )
            li_node = nodes.list_item("")
            li_node += link_node
            enum_list += li_node
        return nodelist


def collect_imagehash_pages(app: Sphinx):
    """Generate pages for each entry in the imagerepo.json"""
    with open(app.config["image_test_json"], "r") as fh:
        imagerepo: ImageRepo = json.load(fh)
    pages = []
    for test, hashfiles in imagerepo.items():
        hashstrs = [hash_from_url(h) for h in hashfiles]
        pages.append(
            (
                f"generated/image_test/{test}",
                {"test": test, "hashfiles": zip(hashstrs, hashfiles)},
                "imagehash.html",
            )
        )
    return pages


def setup(app: Sphinx):
    app.add_config_value(
        "image_test_json",
        "../../lib/iris/tests/results/imagerepo.json",
        "html",
    )

    app.add_directive("imagetest-list", ImageTestDirective)
    app.connect("html-collect-pages", collect_imagehash_pages)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
