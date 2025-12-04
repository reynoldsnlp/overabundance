#!/bin/env python

import os

def generate_index_html(folder):
    items = sorted(os.listdir(folder))
    html_lines = [
        "<html>",
        "<head><title>Index of {}</title></head>".format(os.path.basename(folder)),
        "<body>",
        "<h1>Index of {}</h1>".format(os.path.basename(folder)),
        "<ul>"
    ]
    for item in items:
        if item == "index.html":
            continue
        path = os.path.join(folder, item)
        display = item + "/" if os.path.isdir(path) else item
        html_lines.append(f'<li><a href="{item}">{display}</a></li>')
    html_lines += ["</ul>", "</body>", "</html>"]
    with open(os.path.join(folder, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))

def main():
    docs_dir = "docs"
    for root, dirs, files in os.walk(docs_dir):
        generate_index_html(root)

if __name__ == "__main__":
    main()
