__author__ = 'linked0'
import os
import shutil

filelist = ['./PnTBleFramework_v3/PnTBleFramework/BleSdk/PnTBleService.swift',
    './PnTBleFramework_v3/PnTBleFramework/Common/GeofenceData.swift',
    './PnTBleFramework_v3/PnTBleFramework/Common/GeofencePolygon.swift',
    './PnTBleFramework_v3/PnTBleFramework/Common/Graph.swift',
    './PnTBleFramework_v3/PnTBleFramework/Common/Node.swift',
    './PnTBleFramework_v3/PnTBleFramework/Common/StatusInfo.swift',
    './PnTBleFramework_v3/PnTBleFramework/CouponSdk/BeaconHistoryManager.swift',
    './PnTBleFramework_v3/PnTBleFramework/CouponSdk/BeaconVisitHistory.swift',
    './PnTBleFramework_v3/PnTBleFramework/CouponSdk/HandleScenarios.swift',
    './PnTBleFramework_v3/PnTBleFramework/Navigation/NaviPathInfo.swift',
    './PnTBleFramework_v3/PnTBleFramework/Presence/PnTPresenceMonService.swift',
    './PnTBleFramework_v3/PnTBleFramework/SdkCommon/ContentProvider.swift',
    './PnTBleFramework_v3/PnTBleFramework/SdkCommon/Util.swift',
    './PnTBleFramework_v3/PnTBleFramework/SdkCommon/PnTBleServerSync.swift',
    './PnTBleFramework_v3/PnTBleFramework/SdkCommon/PnTBleKNUHServerSync.swift',
    './PnTBleMap_v3/PnTBleMap/ViewControllers/MapViewController.swift',
    './PnTBleTestAppObjc/PnTBleTestAppObjc/ViewController.m']

sel = raw_input("""
Choose one
1) Xcode6.2 -> Xcode6.3
2) Xcode6.3 -> Xcode6.2
3) Erase All Comments (Use this option after converting to Xcode 6.3 codes)

You Choose: """)

selnum = int(sel)

S63Start = '/*6.3-start*/'
S63End = '/*6.3-end*/'
S62Start = '/*6.2-start*/'
S62End = '/*6.2-end*/'


def convertFileTo63(filename):
    print('convert ' + filename)
    inFile = open(filename)
    outFileName = filename+'.out'
    outFile = open(outFileName, 'w')
    findStart62 = False
    findStart63 = False
    for line in inFile:
        print line
        if line.find(S62Start) != -1:
            findStart62 = True
            print '========> Found 6.2 code'
            outFile.write(line)
            continue
        elif line.find(S62End) != -1:
            findStart62 = False
            outFile.write(line)
            print '========> End 6.2 code'
            continue
        elif line.find(S63Start) != -1:
            findStart63 = True
            print '========> Found 6.3 code'
            outFile.write(line)
            continue
        elif line.find(S63End) != -1:
            findStart63 = False
            outFile.write(line)
            print '========> End 6.3 code'
            continue

        if findStart62 == True and len(line) >= 2 and line[:2] != '//':
            outFile.write('//'+line)
        elif findStart63 == True and len(line) >= 2 and line[:2] == '//':
            outFile.write(line[2:])
        else:
            outFile.write(line)

    inFile.close()
    outFile.close()
    shutil.move(outFileName, filename)

def convertFileTo62(filename):
    print('convert ' + filename)
    inFile = open(filename)
    outFileName = filename+'.out'
    outFile = open(outFileName, 'w')
    findStart62 = False
    findStart63 = False
    for line in inFile:
        print line
        if line.find(S63Start) != -1:
            findStart63 = True
            print '========> Found 6.3 code'
            outFile.write(line)
            continue
        elif line.find(S63End) != -1:
            findStart63 = False
            outFile.write(line)
            print '========> End 6.3 code'
            continue
        elif line.find(S62Start) != -1:
            findStart62 = True
            print '========> Found 6.2 code'
            outFile.write(line)
            continue
        elif line.find(S62End) != -1:
            findStart62 = False
            outFile.write(line)
            print '========> End 6.2 code'
            continue

        if findStart63 == True and len(line) >= 2 and line[:2] != '//':
            outFile.write('//'+line)
        elif findStart62 == True and len(line) >= 2 and line[:2] == '//':
            outFile.write(line[2:])
        else:
            outFile.write(line)

    inFile.close()
    outFile.close()
    shutil.move(outFileName, filename)

def removeComments(filename):
    print('convert ' + filename)
    inFile = open(filename)
    outFileName = filename+'.out'
    outFile = open(outFileName, 'w')
    findStart62 = False
    findStart63 = False
    for line in inFile:
        print line
        if line.find(S63Start) != -1:
            findStart63 = True
            print '========> Found 6.3 code'
            continue
        elif line.find(S63End) != -1:
            findStart63 = False
            print '========> End 6.3 code'
            continue
        elif line.find(S62Start) != -1:
            findStart62 = True
            print '========> Found 6.2 code'
            continue
        elif line.find(S62End) != -1:
            findStart62 = False
            print '========> End 6.2 code'
            continue

        if findStart62 == False:
            outFile.write(line)

    inFile.close()
    outFile.close()
    shutil.move(outFileName, filename)

def convertToLatest():
    print('converting for Xcode 6.3')
    for filename in filelist:
        convertFileTo63(filename)

def convertTo62():
    print('converting for Xcode 6.2')
    for filename in filelist:
        convertFileTo62(filename)

def wrongSelect():
    print('Select 1 or 2 please ^^')

def removeAllComments():
    print('remove all 6.2 comments')
    for filename in filelist:
        removeComments(filename)

if selnum == 1:
    convertToLatest()
elif selnum == 2:
    convertTo62()
elif selnum == 3:
    removeAllComments()
else:
    wrongSelect()
