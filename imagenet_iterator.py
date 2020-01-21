import numpy as np
import os
import cv2
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from queue import Queue
import threading


class ImageNetIterator:

    def __init__(self,path,num_classes,seed,flatten):
        base_path = os.path.join("..","imagenet",path)
        if(not os.path.isdir(base_path)):
            new_base_path = os.path.join("/temp","imagenet",path)
            if(os.path.isdir(new_base_path)):
                base_path = new_base_path

        self.base_path = base_path
        self.flatten = flatten

        all_dirs = sorted([d for d in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path,d))])

        assert len(all_dirs)>0
        rng = np.random.RandomState(seed)
        max_classes = len(all_dirs)
        perm = rng.permutation(max_classes)[:num_classes]

        self.classes = []
        for i in range(perm.shape[0]):
            self.classes.append(all_dirs[perm[i]])
        
        self.synset = {}
        with open("synset.txt","r") as f:
            for line in f:
                key = line[0:9]
                name = line[10:]
                self.synset[key]=name

        self.images = []
        self.labels = []

        self.prefetch = 0

        for i,c in enumerate(self.classes):
            images = sorted([os.path.join(c,f) for f in os.listdir(os.path.join(self.base_path,c)) if f.endswith(".JPEG")])
            self.images.extend(images)

            self.labels.append(i*np.ones(shape=[len(images)],dtype=np.int32))

        self.labels = np.concatenate(self.labels,axis=0)
        print("labels shape: ",str(self.labels.shape))
        print("Total {} images with {} classes loaded".format(self.labels.shape[0],len(self.classes)))

    def size(self):
        return self.labels.shape[0]

    def iterate_with_prefetch(self,batch_size,shuffle=False,prefetch=4):
        if(shuffle):
            perm = np.random.permutation(self.labels.shape[0])
        else:
            perm = None

        number_of_batches = self.size()//batch_size
        parts = np.array_split(np.arange(number_of_batches),prefetch)

        self.batch_queue = Queue(maxsize=prefetch+1)
        
        def task_wrapper_single(part):
            for p in range(part.shape[0]):
                if(self.flatten):
                    batch_x = np.empty([batch_size,224*224*3],dtype=np.float32)
                else:
                    batch_x = np.empty([batch_size,224,224,3],dtype=np.float32)
                batch_y = np.empty([batch_size],dtype=np.int32)
                for i in range(batch_size):
                    read = p*batch_size+i
                    if(shuffle):
                        read = perm[read]

                    image = cv2.imread(os.path.join(self.base_path,self.images[read])).astype(np.float32)
                    assert image.shape[0]==224 and image.shape[1]==224 and image.shape[2]==3, "Assert: Image shape wrong: {} [{}]".format(self.images[read],image.shape)
                    image = image/255.0
                    if(self.flatten):
                        image = image.flatten()
                    batch_x[i] = image
                    batch_y[i] = self.labels[read]
                self.batch_queue.put((batch_x,batch_y))

        threads = [threading.Thread(target=task_wrapper_single, args=(parts[i],)) for i in range(prefetch)]
        for t in threads:
            t.start()

        for b in range(number_of_batches):
            batch_x,batch_y = self.batch_queue.get()
            yield (batch_x,batch_y)

        for t in threads:
            t.join()

    def iterate(self,batch_size,shuffle=False):
        if(shuffle):
            perm = np.random.permutation(self.labels.shape[0])

        if(self.flatten):
            batch_x = np.empty([batch_size,224*224*3],dtype=np.float32)
        else:
            batch_x = np.empty([batch_size,224,224,3],dtype=np.float32)
        batch_y = np.empty([batch_size],dtype=np.int32)

        num_batches = self.size()//batch_size
        for b in range(num_batches):
            for i in range(batch_size):
                read = b*batch_size+i
                if(shuffle):
                    read = perm[read]

                image = cv2.imread(os.path.join(self.base_path,self.images[read])).astype(np.float32)
                assert image.shape[0]==224 and image.shape[1]==224 and image.shape[2]==3, "Assert: Image shape wrong: {} [{}]".format(self.images[read],image.shape)
                image = image/255.0
                if(self.flatten):
                    image = image.flatten()
                batch_x[i] = image
                batch_y[i] = self.labels[read]
            yield (batch_x,batch_y)

def preprocess():
    base_dirs = ["imagenet/training","imagenet/validation"]
    # base_dirs = ["imagenet/training"]
    for base in base_dirs:
        all_dirs = [os.path.join(base,d) for d in os.listdir(base) if os.path.isdir(os.path.join(base,d))]
        # all_dirs = [os.path.join(base,"n02098286")]

        for d in all_dirs:
            print("dir {}".format(d))
            all_files = [f for f in os.listdir(d) if f.endswith(".JPEG")]
            for f in all_files:
                img = cv2.imread(os.path.join(d,f))
                modified = False
                # Make square
                if(img.shape[0] > img.shape[1]):
                    modified = True
                    skip = (img.shape[0]-img.shape[1])//2
                    img = img[skip:skip+img.shape[1]]
                elif(img.shape[1] > img.shape[0]):
                    modified = True
                    skip = (img.shape[1]-img.shape[0])//2
                    img = img[:,skip:skip+img.shape[0]]

                assert img.shape[0]==img.shape[1]
                if(img.shape[0] != 224 or img.shape[1] != 224):
                    img = cv2.resize(img,dsize=(224,224))
                    modified = True
                if(modified):
                    cv2.imwrite(os.path.join(d,f),img,[int(cv2.IMWRITE_JPEG_QUALITY), 95])


if __name__ == "__main__":
    train_iter = ImageNetIterator("imagenet/training",num_classes=2,seed=987,flatten=False)

    for x,y in train_iter.iterate(batch_size=6,shuffle=True):
        break
    
    plt.figure(figsize=(12,4))
    for i in range(x.shape[0]):
        plt.subplot(1,x.shape[0],i+1)
        plt.imshow((255*x[i]).astype(np.uint8))
        plt.title("label: {}\n({})".format(y[i], train_iter.synset[train_iter.classes[int(y[i])]]))
    plt.savefig("test.png")