
from os import listdir
import numpy as np
import cv2

from lib import util

PIXEL_SIZE = 30

untrain = False

class RawData(object):

    def __init__(self, burns):
        self.burns = burns

    @staticmethod
    def load(burnNames='all', dates='all', inference=False):
        print("in rawdata load")
        untrain = False
        if burnNames == 'all':
            print('1')
            burnNames = listdir_nohidden('training_data/')
            print('1')
        if burnNames == 'untrain':
            print('2')
            untrain = True
            burnNames = listdir_nohidden('training_data/_untrained/')
            print('2')
        if dates == 'all':
            print('3')
            burns = {n:Burn.load(n, untrain, 'all', inference=inference) for n in burnNames}
            print('3')
        else:
            # assumes dates is a dict, with keys being burnNames and vals being dates
            print('4')
            burns = {n:Burn.load(n, untrain, dates[n], inference=inference) for n in burnNames}
            print('4')
        return RawData(burns)

    def getWeather(self, burnName, date):
        burn = self.burns[burnName]
        day = burn.days[date]
        return day.weather

    def getOutput(self, burnName, date, location):
        burn = self.burns[burnName]
        day = burn.days[date]
        return day.endingPerim[location]

    def getDay(self, burnName, date):
        return self.burns[burnName].days[date]

    def __repr__(self):
        return "Dataset({})".format(list(self.burns.values()))

class Burn(object):

    def __init__(self, name, days, untrain=False, layers=None):
        self.name = name
        self.days = days
        self._untrain = untrain
        self.layers = layers if layers is not None else self.loadLayers()

        # what is the height and width of a layer of data
        self.layerSize = list(self.layers.values())[0].shape[:2]

    def loadLayers(self):
        folder = 'training_data/{}/'.format(self.name)
        if self._untrain:
            folder = 'training_data/_untrained/{}/'.format(self.name)
        dem = util.openImg(folder+'dem.npy') # tif
        slope = util.openImg(folder+'slope.npy')
        band_2 = util.openImg(folder+'band_2.npy')
        band_3 = util.openImg(folder+'band_3.npy')
        band_4 = util.openImg(folder+'band_4.npy')
        band_5 = util.openImg(folder+'band_5.npy')
        ndvi = util.openImg(folder+'ndvi.npy')
        aspect = util.openImg(folder+'aspect.npy')

        layers = {'dem':dem,
                'slope':slope,
                'ndvi':ndvi,
                'aspect':aspect,
                'band_4':band_4,
                'band_3':band_3,
                'band_2':band_2,
                'band_5':band_5}

        # ok, now we have to make sure that all of the NoData values are set to 0
        #the NV pixels occur outside of our AOIRadius
        # When exported from GIS they could take on a variety of values
        # susceptible = ['dem', 'r','g','b','nir',]
        for name, layer in layers.items():
            pass
        return layers

    @staticmethod
    def load(burnName, untrain=False, dates='all', inference=False):
        # print("Loading: ", burnName)
        # print("in load")
        # print(dates)
        if dates == 'all':
            dates = Day.allGoodDays(burnName, untrain, inference=inference)

        days = {date:Day(burnName, date, untrain, inference=inference) for date in dates}
        return Burn(burnName, days, untrain)

    def __repr__(self):
        return "Burn({}, {})".format(self.name, [d.date for d in self.days.values()])

class Day(object):

    def __init__(self, burnName, date, untrain=False, weather=None, startingPerim=None, endingPerim=None, inference=False):
        self.burnName = burnName
        self.date = date
        self.untrain = untrain
        self.weather = weather             if weather       is not None else self.loadWeather()
        self.startingPerim = startingPerim if startingPerim is not None else self.loadStartingPerim()
        if not inference:
            self.endingPerim = endingPerim     if endingPerim   is not None else self.loadEndingPerim()
        else:
            self.endingPerim = None

    def loadWeather(self):
        fname = 'training_data/{}/weather/{}.csv'.format(self.burnName, self.date)
        if self.untrain:
            fname = 'training_data/_untrained/{}/weather/{}.csv'.format(self.burnName, self.date)

        # the first row is the headers, and only cols 4-11 are actual data
        data = np.loadtxt(fname, skiprows=1, usecols=range(5,12), delimiter=',').T
        # now data is 2D array
        return data

    def loadStartingPerim(self):
        # fname = 'training_data/{}/perims/{}.tif'.format(self.burnName, self.date)
        # perim = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        fname = 'training_data/{}/perims/{}.npy'.format(self.burnName, self.date)
        if self.untrain:
            fname = 'training_data/_untrained/{}/perims/{}.npy'.format(self.burnName, self.date)

        perim = np.load(fname)
        if perim is None:
            raise RuntimeError('Could not find a perimeter for the fire {} for the day {}'.format(self.burnName, self.date))
        # perim[perim!=0] = 255
        perim[perim!=1] = 0
        return perim

    def loadEndingPerim(self):
        guess1, guess2 = Day.nextDay(self.date)
        # fname = 'data/{}/perims/{}.tif'.format(self.burnName, guess1)
        # perim = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        try:
            fname = 'training_data/{}/perims/{}.npy'.format(self.burnName, guess1)
            if self.untrain:
                fname = 'training_data/_untrained/{}/perims/{}.npy'.format(self.burnName, guess1)
            perim = np.load(fname)
        except:
        # if perim is None:
            # overflowed the month, that file didnt exist
            # fname = 'training_data/{}/perims/{}.tif'.format(self.burnName, guess2)
            # perim = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
            fname = 'training_data/{}/perims/{}.npy'.format(self.burnName, guess2)
            if self.untrain:
                fname = 'training_data/_untrained/{}/perims/{}.npy'.format(self.burnName, guess2)
            perim = np.load(fname)
            if perim is None:
                raise RuntimeError('Could not open a perimeter for the fire {} for the day {} or {}'.format(self.burnName, guess1, guess2))


        perim[perim!=1] = 0
        return perim

    def __repr__(self):
        return "Day({},{})".format(self.burnName, self.date)

    @staticmethod
    def nextDay(dateString):
        month, day = dateString[:2], dateString[2:]

        nextDay = str(int(day)+1).zfill(2)
        guess1 = month+nextDay

        nextMonth = str(int(month)+1).zfill(2)
        guess2 = nextMonth+'01'

        return guess1, guess2

    @staticmethod
    def allGoodDays(burnName, untrain=False, inference=False):
        '''Given a fire, return a list of all dates that we can train on (or predict on, if inference=True)'''
        if untrain:
            directory = 'training_data/_untrained/{}/'.format(burnName)
        else:
            directory = 'training_data/{}/'.format(burnName)

        weatherFiles = listdir_nohidden(directory+'weather/')
        weatherDates = [fname[:-len('.csv')] for fname in weatherFiles]

        perimFiles = listdir_nohidden(directory+'perims/')
        perimDates = [fname[:-len('.npy')] for fname in perimFiles if isValidImg(directory+'perims/'+fname)]

        if inference:
            # In inference mode, allow any day with both a perimeter and weather for the current day
            daysWithWeatherAndPerims = [d for d in perimDates if d in weatherDates]
            daysWithWeatherAndPerims.sort()
            return daysWithWeatherAndPerims
        else:
            # we can only use days which have perimeter data on the following day
            daysWithFollowingPerims = []
            for d in perimDates:
                nextDay1, nextDay2 = Day.nextDay(d)
                if nextDay1 in perimDates or nextDay2 in perimDates:
                    daysWithFollowingPerims.append(d)

            # now we have to verify that we have weather for these days as well
            daysWithWeatherAndPerims = [d for d in daysWithFollowingPerims if d in weatherDates]
            daysWithWeatherAndPerims.sort()
            return daysWithWeatherAndPerims

def isValidImg(imgName):
    if imgName.endswith('.npy'):
        try:
            arr = np.load(imgName)
            return arr is not None
        except Exception as e:
            print(f"Failed to load npy file {imgName}: {e}")
            return False
    else:
        img = cv2.imread(imgName, cv2.IMREAD_UNCHANGED)
        return img is not None

def listdir_nohidden(path):
    '''List all the files in a path that are not hidden (begin with a .)'''
    result = []

    for f in listdir(path):
        if not f.startswith('.') and not f.startswith("_"):
            result.append(f)
    return result

if __name__ == '__main__':
    raw = RawData.load()
    print(raw.burns['riceRidge'].days['0731'].weather)
