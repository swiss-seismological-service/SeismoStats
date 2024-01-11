{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      {% if item in members and not item.startswith('_') and not item in inherited_members%}
        ~{{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
      {% if item in members and (not item.startswith('_') or item in ['__call__']) and not item in inherited_members %}
        ~{{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}