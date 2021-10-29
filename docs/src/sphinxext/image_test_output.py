import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import make_refnode
from sphinx.application import Sphinx

ImageRepo = Dict[str, List[str]]

class ImageTestDirective(SphinxDirective):
    def run(self):
        with open(self.config['image_test_json'], 'r') as fh:
            imagerepo = json.load(fh)
        nodelist = []
        for test in imagerepo:
            paragraph_node = nodes.raw(f'<li><a src="/image_test/{test} />{test}</a></li>')
            nodelist.append(paragraph_node)
        return nodelist

def load_imagerepo_json(app: Sphinx, env, added, changed, removed):
    with open(app.config['image_test_json'], 'r') as fh:
        env.imagerepo = json.load(fh)
    return []

def collect_imagehash_pages(app: Sphinx):
    with open(app.config['image_test_json'], 'r') as fh:
        imagerepo: ImageRepo = json.load(fh)
    pages = []
    for test, hashfiles in imagerepo.items():
        hashstrs = []
        pages.append((f'image_test/{test}', {'test': test, 'hashfiles': hashfiles}, 'imagehash.html'))
    return pages


def setup(app: Sphinx):
    app.add_config_value('image_test_json', '../../lib/iris/tests/results/imagerepo.json', 'html')

    # app.add_node(todolist)
    # app.add_node(todo,
    #              html=(visit_todo_node, depart_todo_node),
    #              latex=(visit_todo_node, depart_todo_node),
    #              text=(visit_todo_node, depart_todo_node))

    app.add_directive('imagetest-list', ImageTestDirective)
    # app.add_directive('todolist', TodolistDirective)
    app.connect('env-get-outdated', load_imagerepo_json)
    app.connect('html-collect-pages', collect_imagehash_pages)
    # app.connect('env-merge-info', merge_todos)


    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }