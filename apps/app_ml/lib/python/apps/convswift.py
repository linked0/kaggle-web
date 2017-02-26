import os
import shutil
import re

def calc_line():
    filecount = 0
    total = 0
    listfile = open('./list.txt')
    for line in listfile:
        fileline = 0
        line = line.strip()
        if (len(line) > 0):
            temp = open(line)
            for templine in temp :
                templine = templine.strip()
                if len(templine) > 1 and templine.startswith('//') == False:
                    total = total + 1
                    fileline = fileline + 1
            print line, ":", fileline
            filecount = filecount + 1

    print "total line: " + str(total)
    print "total file: " + str(filecount)
    print "avg line per file: " + str(total/filecount)
    listfile.close()

imports_subst = []

def init_import_st():
    imports_subst.append(('import Foundation', '#import <Foundation/Foundation.h>'))
    imports_subst.append(('import UIKit', '#import <UIKit/UIKit.h>'))
    imports_subst.append(('import CoreLocation', '#import <CoreLocation/CoreLocation.h>'))
    imports_subst.append(('import CoreBluetooth', '#import <CoreBluetooth/CoreBluetooth.h>'))

def get_import_subst(import_st):
    global imports_subst
    for (from_st, to_st) in imports_subst:
        if import_st.find(from_st) == 0:
            return to_st
    return '\n'

def convert_class_def(lines, idx, hfile, mfile):
    classdef = lines[idx]
    li = classdef.split()
    if li[0].find('public') == 0:
        classidx = 2
    else:
        classidx = 1
    hfile.write("\n" + "@interface " + li[classidx]);
    mfile.write("\n\n" + "@implementation " + li[classidx] + "\n\n");
    if li[classidx+1].find("{") == 0:
        hfile.write(": NSObject")
        hfile.write(" // " + classdef)
    hfile.write("\n")
 
    return idx

def convert_var_type(vartype):
    if vartype == "String" or vartype == "String?":
        return "NSString"
    elif vartype == "Int" or vartype == "Int?":
        return "int"
    elif vartype == "Double" or vartype == "Double?":
        return "double"
    else:
        return None;

def re_var_def(varstr):
    # print "re_var_def: " + varstr.strip()

    convvar = None
    mat = None
    onlyvalue = False

    check_varstr = varstr.strip()
    ispublic = False
    if check_varstr.find('public') == 0:
        ispublic = True
        varstr = check_varstr[len('public'):]

    varli = varstr.split()
    ptv = re.compile(r"(?P<mod>let|var) +(?P<name>\w+)\s*:\s*(?P<type>\S+)?\s*=?\s*(?P<value>\w+)?")
    ptvmat = ptv.match(varstr.strip())
    if ptvmat != None:
        mat = ptvmat
       # print "matching ptv"
    else:
        pt = re.compile(r"(?P<mod>let|var) +(?P<name>\S+)\s*:\s*(?P<type>\S+).*")
        ptmat = pt.match(varstr.strip())
        if ptmat != None:
            mat = ptmat
        #  print "matching pt"
        else:
            pv = re.compile(r"(?P<mod>let|var) +(?P<name>\S+) *= *(?P<value>\S+).*")
            pvmat = pv.match(varstr.strip())
            if pvmat != None:
                # print "matching pv"
                onlyvalue = True
                mat = pvmat

    if mat != None and onlyvalue == False:
        convvar = "(nonatomic"
        vartype = mat.group("type")
        convvartype = convert_var_type(vartype)
        if vartype != None and convvartype != None:
            print "vartype: " + vartype + ", conv: " + convvartype
            addst = ""
            if vartype == "String" or vartype == "String?":
                convvar = convvar + ", strong"
                addst = "*"
            convvar = convvar + ") " + convvartype.strip() + " " + addst
        else:
            return None
        
        convvar = convvar +  mat.group("name").strip()
        # print "convvar: " + convvar
        return convvar
    else:
        return None

def convert_var_def(lines, idx, hfile, mfile):
    varstr = lines[idx];
    varli = varstr.split()
    convvar = re_var_def(varstr)
    if convvar != None:
        hfile.write("@property " + convvar + ";")
        hfile.write(" // " + varstr.strip())
    else:
        hfile.write("@property () " + varli[1] + ";")
        hfile.write(" // " + varstr.strip())
    
    hfile.write("\n")
    return idx

def convert_func_def(lines, idx, hfile, mfile):
    funcstr = lines[idx]
   
    funcstr_org = funcstr 
    funcstr = funcstr.strip()
    if funcstr.find('public func') == 0:
        funcstr = funcstr[len('public '):]
    print "Convert Func: " + funcstr.strip()
    
    if funcstr.strip().endswith("{") == False:
        hfile.write(funcstr)
        idx += 1
        return idx

    funcli = funcstr.split()
    startbrace = 0;
    endbrace = 0;
    startidx = idx
    endidx = idx
    while startbrace == 0 or startbrace != endbrace:
        line = lines[idx].strip()
        if line.startswith('//') == False:
            if line.find("{") >= 0: startbrace += 1
            if line.find("}") >= 0: endbrace += 1
        idx += 1;
    endidx = idx

    #print('start: ' + str(startidx) + ', end: ' + str(endidx))
   
    # function info by re
    mat = re.compile(r'(?:public func|func) +(?P<fname>\w+)\((?P<param>\w+\s?:\s?.+)*\)(?: -> )?(?P<rettype>\w+)? {')
    funcinfo = mat.match(funcstr)
    if funcinfo != None:
        info = funcinfo.groupdict()
        print info
        hfile.write("- (")
        if info['rettype'] != None:
            hfile.write(info['rettype'])
        else:
            hfile.write('void')
        hfile.write(')')
        hfile.write(info['fname'])
    else: 
        if funcli[0].find('init') == 0:
            hfile.write("- " + funcli[0])
        else:
            hfile.write("- " + funcli[1])
    hfile.write(" // " + funcstr_org.strip())
    hfile.write("\n")
    
    i = startidx;
    while i < endidx:
        if len(lines[i]) >= 4 and lines[i][0:4] == "    ":
            mfile.write(lines[i][4:])
        else:
            mfile.write(lines[i])
        i += 1

    mfile.write("\n")
    return idx    

def convert_extension_def(lines, idx, hfile, mfile):
    funcstr = lines[idx]
    print "Convert Extension: " + funcstr.strip()

    #if funcstr.strip().endswith("{") == False:
    #    hfile.write(funcstr)
    #    idx += 1
    #    return idx

    funcli = funcstr.split()
    startbrace = 0;
    endbrace = 0;
    startidx = idx
    endidx = idx
    while startbrace == 0 or startbrace != endbrace:
        line = lines[idx].strip()
        if line.startswith('//') == False:
            if line.find("{") >= 0: startbrace += 1
            if line.find("}") >= 0: endbrace += 1
        idx += 1;
    endidx = idx

    #print('start: ' + str(startidx) + ', end: ' + str(endidx))

    # function prototype
    i = startidx;
    while i < endidx:
        mfile.write(lines[i])
        i += 1

    mfile.write("\n")
    return idx
   

def conv(filepath):
    #filelist = [line.strip() for line in open('./list.txt')]
    #idx = filelist.index(name)             

    #print filelist[idx]
    #filepath = filelist[idx]
    init_import_st()

    filename = filepath[filepath.rfind('/')+1:filepath.rfind('.')]
    foldername = filepath[:filepath.rfind('/')]
    print foldername
    print 'filename: ' + filename

    hfilepath = foldername + '/' + filename + '.h'
    mfilepath = foldername + '/' + filename + '.m'
    print hfilepath
    print mfilepath

    swiftfile = open(filepath)
    lines_org = swiftfile.readlines()
    lines = lines_org
    
    #lines = []
    # remove comment line
    #idx = 0
    #for line in lines_org:
    #    if idx < 8:
    #        lines.append(line)
    #    elif line.strip().startswith("//") == False:
    #        lines.append(line)
    #    idx += 1    

    hfile = open(hfilepath, 'w')
    mfile = open(mfilepath, 'w')

    found_firstcomment = False

    idx = 0
    while idx < len(lines):
        # initial comment  
        if found_firstcomment == False and lines[idx].find('//  ' + filename + '.swift') == 0:
            hfile.write("//\n")
            mfile.write("//\n") 
            hfile.write("//  " + filename + ".h\n")
            mfile.write("//  " + filename + ".m\n")
            for i in range(5):
                idx = idx + 1
                hfile.write(lines[idx])
                mfile.write(lines[idx])
            hfile.write("\n")
            mfile.write("\n")
            hfile.write("#ifndef " + filename + "_h\n")
            hfile.write("#define " + filename + "_h\n")

            found_firstcomment = True
         
        # import statement
        elif lines[idx].find('import ') == 0:
            new_import = get_import_subst(lines[idx])
            hfile.write(new_import)
            mfile.write("#import " + "\"" + filename + ".h\"\n")
            mfile.write(new_import)

        # class definition
        elif lines[idx].find('class') == 0 or lines[idx].find('public class') == 0:
            idx  = convert_class_def(lines, idx, hfile, mfile)

        # static variable statement
        elif lines[idx].find("static ") >= 0:
            hfile.write(lines[idx])

        # variable statement (var, let)
        elif lines[idx].find('var ') >= 0 or lines[idx].find('let ') >= 0:
            idx = convert_var_def(lines, idx, hfile, mfile)
        
        # function statement
        elif lines[idx].find('init (') >= 0 or lines[idx].find('init(') >= 0 or lines[idx].find('func ') >= 0 or lines[idx].find('public func ') >= 0:
            idx = convert_func_def(lines, idx, hfile, mfile)

        # extension statement
        elif lines[idx].find('extension ') >= 0:
            idx = convert_extension_def(lines, idx, hfile, mfile)

        else: 
            hfile.write(lines[idx])
        
        idx = idx + 1
    
    hfile.write("\n@end\n")
    mfile.write("\n@end\n")

    hfile.write("#endif /* " + filename + "_h */")
    swiftfile.close()
    hfile.close()
    mfile.close()

def conv_all(files):
    filelist = [line.strip() for line in open(files)]
    for filepath in filelist:
        filepath2 = filepath.strip()
        if filepath2.endswith(".swift") == True and filepath2.startswith("..") == False:
            print filepath
            conv(filepath)

    

    
