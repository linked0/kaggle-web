from bs4 import BeautifulSoup
import pprint

class PoohPrettyPrint(pprint.PrettyPrinter):
    def format(self, _object, context, maxlevels, level):
        if isinstance(_object, unicode):
            return "'%s'" % _object.encode('utf8'), True, False
        elif isinstance(_object, str):
            _object = unicode(_object,'utf8')
            return "'%s'" % _object.encode('utf8'), True, False
        return pprint.PrettyPrinter.format(self, _object, context, maxlevels, level)

def myprint(print_text):
	PoohPrettyPrint().pprint(print_text)
	
# def ex():
#     print 'from pooh.poohprint import mpp'
#     print 'mpp().pprint(result)'
