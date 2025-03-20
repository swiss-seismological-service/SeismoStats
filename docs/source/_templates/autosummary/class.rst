{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}


.. autoclass:: {{ objname }}

   {% block methods %}

   {% block attributes %}

   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      {% if item in members and not item.startswith('_') %}
         {% if (item in inherited_members and objname not in skip_inheritance) or item not in inherited_members %}
         ~{{ name }}.{{ item }}
         {% endif %}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
      {% if item in members and (not item.startswith('_') or item in ['__call__']) %}
         {% if (item in inherited_members and objname not in skip_inheritance) or item not in inherited_members %}
        ~{{ name }}.{{ item }}
        {% endif %}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}