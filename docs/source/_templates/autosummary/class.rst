{{ fullname.replace("seismostats.", "") | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}


   {% block attributes %}

      {% if attributes %}
      .. rubric:: {{ _('Attributes') }}

      .. autosummary::
         :toctree: ../api/
         :nosignatures:
         {% for item in attributes %}
            {% if item in members and not item.startswith('_') %}
               {% if (item in inherited_members and objname not in skip_inheritance) or item not in inherited_members %}
               ~{{ name }}.{{ item }}
               {% endif %}
            {% endif %}
         {%- endfor %}
      {% endif %}
   {% endblock %}

   {% block methods %}
      {% if methods %}
      .. rubric:: {{ _('Methods') }}

      .. autosummary::
         :toctree: ../api/
         :nosignatures:
         {% for item in members %}
            {% if (item in methods or item in ['__call__']) and item not in ['__init__'] %}
               {% if (item in inherited_members and objname not in skip_inheritance) or item not in inherited_members %}
                  ~{{ name }}.{{ item }}
               {% endif %}
            {% endif %}
         {%- endfor %}
      {% endif %}
   {% endblock %}