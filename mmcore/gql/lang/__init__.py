import copy
import json
from mmcore.gql.lang import ast as gqlast
def parse_line(line, target=" "):
    word = ""
    words = []
    for char in list(line + target):
        if char == target:
            if not (word == ""):
                words.append(word)
            word = ""
            continue
        else:
            word += char

    return words


from mmcore.gql.lang import ast as gqlast


def fff(exp):
    match exp:
        case [fld, '{', *expr, '}']:

            ##print(fld, expr, " aaa")
            return gqlast.RootField(gqlast.Field(fld), fff(expr))
        case ['{', *expr, '}']:

            return fff(expr)
        case [ff, rr, '{', *expr, '}', v]:
            d = set([fff([rr, '{', *expr, '}'])])

            d.add(fff(ff))
            d.add(fff(v))

            return d
        case [*w]:
            if len(w) == 1:
                return gqlast.Field(w[0])


            else:
                d = set()
                d.add(fff([w.pop(0)]))
                d.add(fff(w))
                if len(d) > 1:
                    return gqlast.Brackets(*d)
                else:
                    return gqlast.Field(d[0])

        case [w]:

            return fff(w)

        case w:

            ##print("----", w)
            return gqlast.Field(w)


class LineParser:
    replace_map = [("\t", ""),
                   ("\n", ""),
                   ("{", " { "),
                   ("}", " } "),
                   ("(", " ( "),
                   (")", " ) "),
                   (':', " : "),
                   ('"', "")]

    def __init__(self):
        super().__init__()

    def __call__(self, query: str):
        for rpl in self.replace_map:
            query = query.replace(*rpl)

        return fff(parse_line(query))



# language=GraphQL
t = '''
{ 
   objects {
        object {
           name 
           baz {
             uuid
           }
        } 
     }
  }'''
# parse_line(t)
# Out[9]: ['query', 'ObjectTestQuery', '{', 'object', '{', 'name', 'uuid', '}', '}']
data_test = {'ObjectTestQuery': {'object': {'name': "foo", "uuid": "bar", "baz": {'name': "A", "uuid": "1111"}}}}

