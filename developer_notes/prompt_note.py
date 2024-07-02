def python_str_format_example(task_desc_str: str, input_str: str):

    # percent(%) formatting
    print("<SYS>%s</SYS> User: %s" % (task_desc_str, input_str))

    # format() method with kwargs
    print(
        "<SYS>{task_desc_str}</SYS> User: {input_str}".format(
            task_desc_str=task_desc_str, input_str=input_str
        )
    )

    # f-string
    print(f"<SYS>{task_desc_str}</SYS> User: {input_str}")

    # Templates
    from string import Template

    t = Template("<SYS>$task_desc_str</SYS> User: $input_str")
    print(t.substitute(task_desc_str=task_desc_str, input_str=input_str))


def jinja2_template_example(template, **kwargs):
    from jinja2 import Template

    t = Template(template, trim_blocks=True, lstrip_blocks=True)
    print(t.render(**kwargs))


def lightrag_prompt(template, task_desc_str, input_str, tools=None):
    from lightrag.core.prompt_builder import Prompt

    prompt = Prompt(
        template=template,
        prompt_kwargs={
            "task_desc_str": task_desc_str,
            "tools": tools,
        },
    )
    print(prompt)
    print(prompt(input_str=input_str))

    saved_prompt = prompt.to_dict()
    restored_prompt = Prompt.from_dict(saved_prompt)
    print(
        restored_prompt == prompt
    )  # False as the jinja2 template can not be serialized, but we recreated the template from the string at the time of restoration, so it works the same
    print(restored_prompt)


def lightrag_default_prompt():
    from lightrag.core.prompt_builder import Prompt

    prompt = Prompt()
    input_str = "What is the capital of France?"
    output = prompt(input_str=input_str)
    print(output)


if __name__ == "__main__":

    task_desc_str = "You are a helpful assitant"
    input_str = "What is the capital of France?"
    tools = ["google", "wikipedia", "wikidata"]
    template = r"""<SYS>{{ task_desc_str }}</SYS>
{# tools #}
{% if tools %}
<TOOLS>
{% for tool in tools %}
{{loop.index}}. {{ tool }}
{% endfor %}
</TOOLS>
{% endif %}
User: {{ input_str }}"""
    python_str_format_example(task_desc_str, input_str)
    jinja2_template_example(template, task_desc_str=task_desc_str, input_str=input_str)
    jinja2_template_example(
        template, task_desc_str=task_desc_str, input_str=input_str, tools=tools
    )
    lightrag_prompt(
        template, task_desc_str=task_desc_str, input_str=input_str, tools=tools
    )

    lightrag_default_prompt()
