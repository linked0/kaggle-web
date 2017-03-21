### 진도
- assign1.py에서 D 문자 이미지까지 pickle로 정리함.

### Assignment1 Problem6 
- 코드 및 결과
pickle_file = 'notMNIST2.pickle'
try:
    f = open(pickle_file, 'rb')
    data = pickle.load(f)
except Exception as e:
    print('unabel to load data from ', pickle_file, ':', e)

train_dataset = data['train_dataset']
train_labels = data['train_labels']
valid_dataset_clean = data['valid_dataset']
valid_labels_clean = data['valid_labels']
test_dataset_clean = data['test_dataset']
test_labels_clean = data['test_labels']

ted = as1.test_dataset_clean
tel = as1.test_labels_clean
ted = ted.reshape((ted.shape[0], ted.shape[1]*ted.shape[2]))
ted_part = ted[:50]
tel_part = tel[:50]

trd_50000 = trd[:50000, :]
trl_50000 = trl[:50000]
In [176]: clf.fit(trd_50000, trl_50000)
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:  3.0min finished
Out[176]: 
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=1000, multi_class='multinomial',
          n_jobs=1, penalty='l2', random_state=42, solver='lbfgs',
          tol=0.0001, verbose=1, warm_start=False)

In [177]: clf.score(ted_part, tel_part)
Out[177]: 0.92000000000000004

clf.predict(ted_part)
Out[195]: 
array([8, 2, 3, 1, 0, 7, 5, 6, 7, 4, 3, 0, 7, 3, 7, 4, 3, 3, 9, 1, 9, 0, 3,
       2, 1, 3, 0, 8, 6, 5, 4, 1, 9, 3, 3, 7, 9, 8, 1, 3, 9, 7, 7, 4, 8, 5,
       2, 8, 0, 5], dtype=int32)

In [204]: tel_part
Out[204]: 
array([8, 2, 3, 1, 0, 7, 5, 6, 8, 4, 3, 0, 7, 3, 7, 4, 3, 3, 9, 1, 9, 0, 3,
       2, 1, 3, 0, 8, 6, 5, 4, 1, 9, 3, 1, 7, 9, 8, 1, 3, 9, 7, 7, 4, 8, 5,
       2, 9, 0, 2], dtype=int32)


###########################################################################


Assignments for Udacity Deep Learning class with TensorFlow
===========================================================

Course information can be found at https://www.udacity.com/course/deep-learning--ud730

Running the Docker container from the Google Cloud repository
-------------------------------------------------------------

    docker run -p 8888:8888 -it --rm b.gcr.io/tensorflow-udacity/assignments

Accessing the Notebooks
-----------------------

On linux, go to: http://127.0.0.1:8888

On mac, find the virtual machine's IP using:

    docker-machine ip default

Then go to: http://IP:8888 (likely http://192.168.99.100:8888)

FAQ
---

* **I'm getting a MemoryError when loading data in the first notebook.**

If you're using a Mac, Docker works by running a VM locally (which
is controlled by `docker-machine`). It's quite likely that you'll
need to bump up the amount of RAM allocated to the VM beyond the
default (which is 1G).
[This Stack Overflow question](http://stackoverflow.com/questions/32834082/how-to-increase-docker-machine-memory-mac)
has two good suggestions; we recommend using 8G.

In addition, you may need to pass `--memory=8g` as an extra argument to
`docker run`.

Notes for anyone needing to build their own containers (mostly instructors)
===========================================================================

Building a local Docker container
---------------------------------

    cd tensorflow/examples/udacity
    docker build -t $USER/assignments .

Running the local container
---------------------------

To run a disposable container:  

    docker run -p 8888:8888 -it --rm $USER/assignments

Note the above command will create an ephemeral container and all data stored in the container will be lost when the container stops.

To avoid losing work between sessions in the container, it is recommended that you mount the `tensorflow/examples/udacity` directory into the container:

    docker run -p 8888:8888 -v </path/to/tensorflow/examples/udacity>:/notebooks -it --rm $USER/assignments

This will allow you to save work and have access to generated files on the host filesystem.

Pushing a Google Cloud release
------------------------------

    V=0.1.0
    docker tag $USER/assignments b.gcr.io/tensorflow-udacity/assignments:$V
    docker tag $USER/assignments b.gcr.io/tensorflow-udacity/assignments:latest
    gcloud docker push b.gcr.io/tensorflow-udacity/assignments
