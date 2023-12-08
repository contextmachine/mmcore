# include <stdint.h>
# include <string.h>

struct MeshTuple{
    char** attributes;
    uint32_t* indices;
    char** extras;
};

struct extras_dict{
    char* children;
};

char* union_mesh(struct MeshTuple meshes[], const struct extras_dict extras)
{
    struct extras_dict ex;
    if (!(extras.children)) {
        ex = (struct extras_dict) {.children = ""};
    } else {
        ex = extras;
    }

    char** attribute_names = _get_attribute_names(meshes);

    uint32_t* indices_and_extras = calloc(100, sizeof(uint32_t));
    int len = sizeof (attribute_names) / sizeof (attribute_names[0]);
    for (int i = 0; i < len; i++)
    {
        indices_and_extras = gen_indices_and_extras2(meshes, attribute_names, i);
    }

    char** attributes = _combine_attributes(indices_and_extras, attribute_names);
    uint32_t* indices = _get_indices(indices_and_extras);
    char* children = _get_children(indices_and_extras);

    ex.children = children;

    struct MeshTuple result;
    result.attributes = attributes;
    result.indices = indices;
    result.extras = ex.children;

    return result;
}

char** _get_attribute_names(struct MeshTuple *meshes)
{
    char** names = extract_mesh_attrs_union_keys(meshes);
    int len = sizeof (names) / sizeof (names[0]);
    if (MESH_OBJECT_ATTRIBUTE_NAME not in names) {
        for (int i = 0; i < len; i++) {
            names[i] = MESH_OBJECT_ATTRIBUTE_NAME;
        }
    }

    return names;
}

char* _combine_attributes(uint32_t* indices_and_extras, char** attribute_names)
{
    char** result = calloc(100, sizeof(char));
    int len = sizeof (attribute_names) / sizeof (attribute_names[0]);

    for (int i = 0; i < len; i++)
    {
        strcat(result[i], indices_and_extras);
    }

    return result;
}

uint32_t* _get_indices(uint32_t* indices_and_extras)
{
    return indices_and_extras;
}

char* _get_children(uint32_t* indices_and_extras)
{
    return indices_and_extras[(sizeof(indices_and_extras) / sizeof(indices_and_extras[0]))-1];
}

struct MeshTuple sum_meshes(struct MeshTuple a, struct MeshTuple b)
{
    return union_mesh(a, b);
}

struct MeshTuple MeshTuple_add(struct MeshTuple a, struct MeshTuple b)
{
    return sum_meshes(a, b);
}

// TODO: Implement the rest of the methods...