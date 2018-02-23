
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread, imresize, imsave
#from scipy.misc import
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import glob
import shutil
import hashlib

os.chdir('/Users/arielkelman/Documents/Ariel/EngSci3-PhysicsOption/Winter2018/CSC411 - Machine Learning/Project2/CSC411/')



f = 'resources/facescrub_actresses.txt'
save_to = 'resources/croppedFemale_p10/' #directory for saving images
gender = 'female' #for saving downloaded and skipped records

f = 'resources/facescrub_actors.txt'
save_to = 'resources/croppedMale_p10/'
gender = 'male'

act = list(set([a.split("\t")[0] for a in open(f).readlines()])) #"subset_actors.txt"
#act = ['Lorraine Bracco', 'Angie Harmon', 'Peri Gilpin']
act = ['Alec Baldwin', 'Bill Hader', 'Steve Carell']

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.request.URLopener()


#Create & empty the folder to save images
if os.path.exists(save_to):
    shutil.rmtree(save_to)
os.makedirs(save_to)


#define blacklisted images (unable to download these); lists carryover from Project 1
blacklist_male = ['http://www.hairstyleshaircut.net/photos/gerard-butler.jpg', 'http://collegepals.org/wp-content/uploads/2012/07/220px-Gerard_Butler_Berlin_Film_Festival_2011.jpg', 'http://laslice.com/wp-content/uploads/2011/04/Alec_Baldwin.jpg', 'http://www.releasedonkey.com/big/MV5BMjI2NjYyOTU4MV5BMl5BanBnXkFtZTcwOTMyMTIwOQ/alec-baldwin-large-picture.jpg', 'http://www.celebrityschoolpics.com/images/g_full/celebrity-001819-gerard-butler.jpg', 'http://www.celebritiesheight.com/wp-content/uploads/2011/07/Alec-Baldwin.jpg', 'http://www.releasedonkey.com/big/MV5BOTMwMDcyNjk0M15BMl5BanBnXkFtZTcwNDUxOTMwNw/bill-hader-large-picture.jpg', 'http://www.releasedonkey.com/big/MV5BMjQwNjEwNjMyN15BMl5BanBnXkFtZTcwODc5MTAyNw/picture-of-bill-hader-in-forgetting-sarah-marshall-2008--large-picture.jpg', 'http://godcelebs.com/images/bill-hader/bill-hader.jpg', 'http://cinesilver.com/wp-content/uploads/2012/04/Steve-Carell-joins-Warner-Bros..jpg', 'http://worldhdwallpaper.com/wp-content/uploads/2013/04/daniel-radcliffe-hd-wallpapers.jpg', 'http://worldhdwallpaper.com/wp-content/uploads/2013/04/Daniel-Radcliffe-cute-face-hd-images.jpeg', 'http://www.moviestarspicture.com/photos/daniel-radcliffe-wallpaper/daniel-radcliffe-latest-wallpaper-2013.jpg', 'http://www.moviestarspicture.com/photos/daniel-radcliffe-wallpaper/daniel-radcliffe-as-harry-potter-wallpaper.jpg']
blacklist_female = ['http://www.angieharmon.net/angie-harmon-pictures/cache/events-parties/angie-harmon-cleavage-shiny_595.jpg', 'http://searchweight.com/wp-content/uploads/2012/01/angie-harmon-2.jpg', 'http://www.hairstylestime.com/images/angie-harmon-hairstyles/Angie-Harmon-1.jpg', 'http://verbalslap.com/wp-content/uploads/2013/08/Angie-Harmon-featured.jpg', 'http://www.hairstylestime.com/images/angie-harmon-hairstyles/Angie-Harmon2333.jpg', 'http://www.bestcelebwallpapers.com/wallpapers/1622-angie-harmon-1024x768.jpg', 'http://measurements.matchincome.com/wp-content/uploads/2013/11/Fran-Drescher-Actress.jpg', 'http://www.celebritiesheight.com/wp-content/uploads/2011/10/Fran-Drescher.jpg', 'http://www.momentumwomen.com/files/momentumwomen/may2008/SteveStory/24-CeslieFranDrescher.jpg', 'http://elisabeth.at/fotos/fran-drescher-1.jpg', 'http://www.seupeso.com/images/l/Lorraine_Bracco.jpg', 'http://fcovers.net/covers/lorraine_bracco-851x315.jpg', 'http://www.celebritiesheight.com/wp-content/uploads/2012/09/Lorraine-Bracco-210x300.jpg', 'http://www.goldenstateautographs.com/New-Arrivals/images/braccolorraine.jpg', 'http://measurements.matchincome.com/wp-content/uploads/2013/12/Peri-Gilpin-image.jpg', 'http://images.movieplayer.it/2012/04/09/peri-gilpin-236972.jpg', 'http://www.releasedonkey.com/big/MV5BMTc3NzE4NTIzMV5BMl5BanBnXkFtZTcwMjkyMzgxNA/picture-of-kelsey-grammer-david-hyde-pierce-john-mahoney-peri-gilpin-and-jane-leeves-in-frasier-1993--large-picture.jpg', 'http://www.releasedonkey.com/big/MV5BMzYyNTAyMDAzNF5BMl5BanBnXkFtZTcwMzMwNjk0OA/frank-grillo-jake-gyllenhaal-anna-kendrick-michael-pe-a-america-ferrera-natalie-martinez-and-cody-horn-in-end-of-watch-2012--large-picture.jpg', 'http://bstylish.info/wp-content/uploads/2013/10/purple-peplum-topamerica-ferrera-wearing-purple-peplum-top-rrkge4pk.jpg', 'http://images.newcelebritypics.com/img/celebs/images/a/america_ferrera_headshot-3984.jpg', 'http://www.releasedonkey.com/big//MV5BMTQyMDUzOTkxNF5BMl5BanBnXkFtZTcwNjMwNjk0OA/america-ferrera-large-picture.jpg', 'http://searchweight.com/wp-content/uploads/2011/12/america-ferrera-3.jpg', 'http://www.afashionhub.com/wp-content/uploads/2011/02/America-Ferrera-Long-Hairstyle.jpg', 'http://www.decisivelatino.com/sites/default/files/America-Ferrera.jpg', 'http://www.hairstylestime.com/images/kristin-chenoweth-hairstyles/kristin-chenowet-282401.jpg', 'http://www.1920x1200.net/posts/wp-content/uploads/2012/09/kristin_chenoweth_1920_1200_sep222012.jpg', 'http://picture.fm/wp-content/uploads/2013/06/kristin-chenoweth_d5I1poe6n7.jpg']
blacklist = blacklist_male + blacklist_female

print(act) #print actors for whom data will be downloaded
BUF_SIZE = 65536
sz = (227,227) #size to resize images to; (32,32) for earlier parts

skipped = '' #will contain all the URLs of images not downloaded
no_skipped = 0
bad_hash = ''
no_hashed = 0
downloaded = ''
no_downloaded = 0

for a in act:
    name = a.split()[1].lower()
    i = 0
    for line in open(f): #"faces_subset.txt"
        if a in line and line.split()[4] not in blacklist:
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                #A version without timeout (uncomment in case you need to
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long to download
            timeout(testfile.retrieve, (line.split()[4], save_to+filename), {}, 60)
            
            if os.path.isfile(save_to+filename):
                sha256 = hashlib.sha256() #reset the hash
                with open(save_to+filename, 'rb') as m: #most of this code comes from stackoverflow
                    while True:
                        data = m.read(BUF_SIZE)
                        if not data:
                            break
                        sha256.update(data)
                hash = sha256.hexdigest()
                correct_hash = line.split()[6]
                
                if hash == correct_hash:
                    print(filename)
                    #print(line.split()[4])
                    downloaded += filename + ' ' + line.split()[4] + '\n'
                    no_downloaded += 1
                    
                    img = imread(save_to+filename, flatten = False) #flatten = True for grayscale
                    
                    x1,y1,x2,y2 = [int(k) for k in line.split()[5].split(',')] #bounding box

                    img = img[y1:y2, x1:x2] 
                    img = imresize(img, sz)
                    
                    if False and i<1:
                        plt.imshow(img) #, cmap = cm.gray
                        plt.show()
                        print( np.shape(img) )
                    
                    imsave(save_to+filename, img)
                    i += 1
                else:
                    os.remove(save_to+filename)
                    print('Error: incorrect hash - hash: ' + hash)
                    bad_hash += name + ' ' + line.split()[4] + '\n'
                    no_hashed += 1
            else:
                print('Error: missing file')
                skipped += name + ' ' + line.split()[4] + '\n'
                no_skipped += 1


print("Downloaded: " + str(no_downloaded) )
print("Skipped: " + str(no_skipped) )
print("Bad Hash: " + str(no_hashed) )

g = open('resources/skipped_' + gender + '.txt', 'w+') #File containing a list of skipped files and URLs
g.write('Total Skipped: ' + str(no_skipped) + '\n')
g.write(skipped)
g.close()
h = open('resources/downloaded_' + gender + '.txt', 'w+') #File containing a list of downloaded files and URLs
h.write('Total Downloaded: ' + str(no_downloaded) + '\n')
h.write(downloaded)
h.close()
k = open('resources/hashed_' + gender + '.txt', 'w+') #File containing a list of downloaded files and URLs
k.write('Total Bad Hashes: ' + str(no_hashed) + '\n')
k.write(bad_hash)
k.close()
