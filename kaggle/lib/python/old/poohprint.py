from bs4 import BeautifulSoup
import pprint

class mpp(pprint.PrettyPrinter):
    def format(self, _object, context, maxlevels, level):
        if isinstance(_object, unicode):
            return "'%s'" % _object.encode('utf8'), True, False
        elif isinstance(_object, str):
            _object = unicode(_object,'utf8')
            return "'%s'" % _object.encode('utf8'), True, False
        return pprint.PrettyPrinter.format(self, _object, context, maxlevels, level)

def example():
    print 'from pooh import mpp'
    print 'mpp().pprint(result)'
