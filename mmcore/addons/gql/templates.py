_query_temp = """
query {
  {{ root }} {
    {% for attr in attrs %}
    {{ attr }}{% endfor %}
  }
}
"""
_mutation_insert_one = """
mutation {{ directive_name }} ({{ x }}) {
     insert_{{ schema }}_{{ table }}_one(object: {{ y }}) {
        {% for attr in attrs %}
        {{ attr }}{% endfor %}
  }
}
"""
