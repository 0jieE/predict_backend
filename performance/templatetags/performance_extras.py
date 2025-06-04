from django import template

register = template.Library()

@register.filter
def index(sequence, position):
    try:
        return sequence[position]
    except:
        return ''

@register.filter
def replace_underscore(value):
    return value.replace("_", " ")

@register.filter
def faculty_shortname(name):
    try:
        last, first = [part.strip() for part in name.split(",")]
        return f"Faculty {first[0]}. {last}"
    except:
        return name  # fallback if unexpected format