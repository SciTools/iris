{% if render_title %}
{% set months = {
    "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
    "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
    "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec"
} %}
{% set parts = versiondata.date.split("-") %}
{% set date = parts[2] + " " + months[parts[1]] + " " + parts[0] %}
{% set qualifier = " [unreleased]" if "dev" in versiondata.version else "" %}
{% set extra = qualifier|length %}
{% set parts = versiondata.version.split(".") %}
{% set release = parts[0] + "." + parts[1] %}
{% set version = versiondata.version if "dev" in versiondata.version else release %}
v{{ version }} ({{ date }}){{ qualifier }}
{{ top_underline * ((version + date)|length + 4 + extra)}}
{% endif %}
{% for section, _ in sections.items() %}
{% set underline = underlines[1] %}

.. include:: highlights.rst

{% if sections[section] %}
{% for category, val in definitions.items() if category in sections[section]%}
{{ definitions[category]['name'] }}
{{ underline * (definitions[category]['name']|length + 1) }}

{% for text, values in sections[section][category]|dictsort(by='value') %}
{% set comma = joiner(', ') %}
- {% for value in values|sort %}{{ comma() }}:fa:`code-pull-request` :pull:`{{ value[1:] }}`{% endfor %}: {{ text|replace(":issue:", ":far:`circle-dot` :issue:")|replace(":pull:", ":fa:`code-pull-request` :pull:") }}

{% endfor %}

{% if sections[section][category]|length == 0 %}
No significant changes.

{% else %}
{% endif %}

{% endfor %}
{% else %}
No significant changes.


{% endif %}
{% endfor %}
