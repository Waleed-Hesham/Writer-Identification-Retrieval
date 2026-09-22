import os
import shlex
import argparse
from tqdm import tqdm

# for python3: read in python2 pickled files
import _pickle as cPickle

import gzip
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
import numpy as np
import cv2
from parmap import parmap
from reranking import rerank_distances
import sys
from sklearn.decomposition import PCA

def parseArgs(parser):
    parser.add_argument('--labels_test', 
                        help='contains test images/descriptors to load + labels')
    parser.add_argument('--labels_train', 
                        help='contains training images/descriptors to load + labels')
    parser.add_argument('-s', '--suffix',
                        default='_SIFT_patch_pr.pkl.gz',
                        help='only chose those images with a specific suffix')
    parser.add_argument('--in_test',
                        help='the input folder of the test images / features')
    parser.add_argument('--in_train',
                        help='the input folder of the training images / features')
    parser.add_argument('--overwrite', action='store_true',
                        help='do not load pre-computed encodings')
    parser.add_argument('--powernorm', action='store_true',
                        help='use powernorm')
    parser.add_argument('--gmp', action='store_true',
                        help='use generalized max pooling')
    parser.add_argument('--gamma', default=1, type=float,
                        help='regularization parameter of GMP')
    parser.add_argument('--C', default=1000, type=float, 
                        help='C parameter of the SVM')
    
    parser.add_argument('--use_images', action='store_true',
                        help='bonus (e): use original images + compute SIFT(Hellinger) instead of loading .pkl.gz')
    parser.add_argument('--multivlad', action='store_true',
                        help='bonus (g): use 5 codebooks + concatenate VLAD + PCA whitening (1000D)')
    parser.add_argument('--n_codebooks', default=5, type=int,
                        help='number of codebooks for multi-VLAD (default 5)')
    parser.add_argument('--pca_dim', default=1000, type=int,
                        help='PCA output dimension for multi-VLAD (default 1000)')
    parser.add_argument('--pca_subset', default=5000, type=int,
                        help='max number of training samples used to fit PCA (speed/memory)')
    parser.add_argument('--k', default=100, type=int,
                        help='number of clusters (K) (default 100; you may reduce to e.g. 32 for speed)')

    parser.add_argument('--rerank', default='qe+kreciprocal',
                        choices=['none', 'qe', 'kreciprocal', 'qe+kreciprocal'],
                        help='re-ranking method applied after the encodings are computed')
    parser.add_argument('--qe_k', default=2, type=int,
                        help='number of neighbors used for alpha query expansion')
    parser.add_argument('--qe_alpha', default=3.0, type=float,
                        help='weighting exponent of alpha query expansion')
    parser.add_argument('--qe_iter', default=1, type=int,
                        help='number of query expansion iterations')
    # defaults tuned for ICDAR17: only 4 relevant pages per query, so k1 must stay small
    parser.add_argument('--rerank_k1', default=4, type=int,
                        help='k1 neighborhood size of the k-reciprocal re-ranking')
    parser.add_argument('--rerank_k2', default=2, type=int,
                        help='k2 local expansion size of the k-reciprocal re-ranking')
    parser.add_argument('--rerank_lambda', default=0.3, type=float,
                        help='weight of the original distance vs. the Jaccard distance')
    return parser

def getFiles(folder, pattern, labelfile):
    """ 
    returns files and associated labels by reading the labelfile 
    parameters:
        folder: inputfolder
        pattern: new suffix
        labelfiles: contains a list of filename and labels
    return: absolute filenames + labels 
    """
    # read labelfile
    with open(labelfile, 'r') as f:
        all_lines = f.readlines()
    
    # get filenames from labelfile
    all_files = []
    labels = []
    check = True
    for line in all_lines:
        # using shlex we also allow spaces in filenames when escaped w. ""
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

        # strip all known endings, note: os.path.splitext() doesnt work for
        # '.' in the filenames, so let's do it this way...
        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.tif', '.ocvmb','.csv']:
            if file_name.endswith(p):
                file_name = file_name.replace(p,'')

        # get now new file name
        true_file_name = os.path.join(folder, file_name + pattern)
        if not os.path.exists(true_file_name):
            # try .jpg first, then .png
            jpg_name = os.path.join(folder, file_name + '.jpg')
            png_name = os.path.join(folder, file_name + '.png')
            if os.path.exists(jpg_name):
                true_file_name = jpg_name
            elif os.path.exists(png_name):
                true_file_name = png_name

        all_files.append(true_file_name)
        labels.append(class_id)

    return all_files, labels

def loadRandomDescriptors(files, max_descriptors):
    """ 
    load roughly `max_descriptors` random descriptors
    parameters:
        files: list of filenames containing local features of dimension D
        max_descriptors: maximum number of descriptors (Q)
    returns: QxD matrix of descriptors
    """
    # let's just take 100 files to speed-up the process
    max_files = 100
    indices = np.random.permutation(max_files)
    files = np.array(files)[indices]
   
    # rough number of descriptors per file that we have to load
    max_descs_per_file = int(max_descriptors / len(files))

    descriptors = []
    for i in tqdm(range(len(files))):
        with gzip.open(files[i], 'rb') as ff:
            # for python2
            # desc = cPickle.load(ff)
            # for python3
            desc = cPickle.load(ff, encoding='latin1')
            
        # get some random ones
        indices = np.random.choice(len(desc),
                                   min(len(desc),
                                       int(max_descs_per_file)),
                                   replace=False)
        desc = desc[ indices ]
        descriptors.append(desc)
    
    descriptors = np.concatenate(descriptors, axis=0)
    return descriptors

def dictionary(descriptors, n_clusters):
    """ 
    return cluster centers for the descriptors 
    parameters:
        descriptors: NxD matrix of local descriptors
        n_clusters: number of clusters = K
    returns: KxD matrix of K clusters
    """
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=1000,
        verbose=False,
        compute_labels=False
    )
    kmeans.fit(descriptors)
    mus = kmeans.cluster_centers_
    return mus
def assignments(descriptors, clusters):
    """ 
    compute assignment matrix
    parameters:
        descriptors: TxD descriptor matrix
        clusters: KxD cluster matrix
    returns: TxK assignment matrix
    """
    # compute nearest neighbors

    bf = cv2.BFMatcher(cv2.NORM_L2)
    # BFMatcher expects float32
    desc_f32 = descriptors.astype(np.float32)
    clus_f32 = clusters.astype(np.float32)

    if desc_f32.ndim != 2 or clus_f32.ndim != 2:
        raise ValueError('Descriptors and clusters must be two-dimensional matrices.')
    if desc_f32.shape[1] != clus_f32.shape[1]:
        raise ValueError(
            'Descriptor dimension ({}) does not match cluster dimension ({}). '
            'Rebuild the dictionary for the selected descriptor pipeline.'.format(
                desc_f32.shape[1], clus_f32.shape[1]))

    # knnMatch with k=1 to get nearest cluster for each descriptor
    matches = bf.knnMatch(desc_f32, clus_f32, k=1)

    # create hard assignment
    assignment = np.zeros((len(descriptors), len(clusters)), dtype=np.float32)
    for i, mlist in enumerate(matches):
        m = mlist[0]           # best match
        k = m.trainIdx         # index of nearest cluster
        assignment[i, k] = 1.0

    return assignment

def vlad(files, mus, powernorm, gmp=False, gamma=1000):
    """
    compute VLAD encoding for each files
    parameters: 
        files: list of N files containing each T local descriptors of dimension
        D
        mus: KxD matrix of cluster centers
        gmp: if set to True use generalized max pooling instead of sum pooling
    returns: NxK*D matrix of encodings
    """
    K = mus.shape[0]
    encodings = []

    for f in tqdm(files):
        with gzip.open(f, 'rb') as ff:
            desc = cPickle.load(ff, encoding='latin1')
        a = assignments(desc, mus)
        
        T,D = desc.shape
        f_enc = np.zeros( (D*K), dtype=np.float32)
        for k in range(mus.shape[0]):
            # it's faster to select only those descriptors that have
            # this cluster as nearest neighbor and then compute the 
            # difference to the cluster center than computing the differences
            # first and then select
                        # indices of descriptors assigned to cluster k
            idx = np.where(a[:, k] == 1)[0]
            if len(idx) == 0:
                # nothing assigned to this cluster -> leave zeros
                continue

            # descriptors assigned to this cluster
            desc_k = desc[idx, :]                       # shape: (T_k, D)
            residuals = desc_k - mus[k][None, :]        # x - μ_k

            if not gmp:
                # standard VLAD: sum pooling over residuals
                v_k = residuals.sum(axis=0)             # shape: (D,)
            else:
                # GMP: solve ridge regression component-wise
                # X = residuals, y = 1 vector
                y = np.ones(residuals.shape[0], dtype=np.float32)
                rr = Ridge(
                    alpha=gamma,
                    fit_intercept=False,
                    solver='sparse_cg',
                    max_iter=500
                )
                rr.fit(residuals, y)
                # coef_ is the encoding for this cluster (D-dimensional)
                v_k = rr.coef_.ravel()

            # put v_k into the right position in f_enc
            start = k * D
            end = (k + 1) * D
            f_enc[start:end] = v_k.astype(np.float32)

   
        # c) power normalization
        if powernorm:
            # signed square-root: xi' = sign(xi) * |xi|^0.5
            f_enc = np.sign(f_enc) * np.sqrt(np.abs(f_enc))

        # l2 normalization 
        f_enc = normalize(f_enc.reshape(1, -1), norm='l2')[0]

        encodings.append(f_enc)

    encodings = np.vstack(encodings)
    return encodings

def esvm(encs_test, encs_train, C=1000):
    """ 
    compute a new embedding using Exemplar Classification
    compute for each encs_test encoding an E-SVM using the
    encs_train as negatives   
    parameters: 
        encs_test: NxD matrix
        encs_train: MxD matrix

    returns: new encs_test matrix (NxD)
    """


    # set up labels

    neg_labels = -np.ones(encs_train.shape[0], dtype=np.int32)

    def loop(i):
        # compute SVM 
        # and make feature transformation
        # positive sample: encs_test[i]
        x_pos = encs_test[i].reshape(1, -1)          # 1xD
        # negatives: all encs_train
        X = np.vstack((x_pos, encs_train))          # (1+M) x D

        # labels: +1 for positive, -1 for all negatives
        y = np.concatenate(([1], neg_labels))

        # compute SVM 
        svm = LinearSVC(C=C, class_weight='balanced')
        svm.fit(X, y)

        # weight vector is the new feature
        w = svm.coef_.reshape(1, -1)                # 1xD
        # l2-normalize w
        w_norm = normalize(w, norm='l2')

        # return as 1xD so parmap can stack them
        return w_norm
        # return x

    # let's do that in parallel: 
    # if that doesn't work for you, just exchange 'parmap' with 'map'
    # Even better: use DASK arrays instead, then everything should be
    # parallelized
    new_encs = list(parmap( loop, tqdm(range(len(encs_test)))))
    new_encs = np.concatenate(new_encs, axis=0)
    # return new encodings
    return new_encs


def distances(encs):
    """ 
    compute pairwise distances 

    parameters:
        encs:  TxK*D encoding matrix
    returns: TxT distance matrix
    """
    # compute cosine distance = 1 - dot product between l2-normalized
    # encodings
    # l2-normalize each encoding
    encs_norm = normalize(encs, norm='l2', axis=1)

    # cosine similarity is dot product of normalized vectors
    sim = np.dot(encs_norm, encs_norm.T)   # TxT

    # cosine distance = 1 - similarity
    dists = 1.0 - sim

    # mask out distance with itself
    np.fill_diagonal(dists, np.finfo(dists.dtype).max)
    return dists

def evaluate(encs, labels, dist_matrix=None):
    """
    evaluate encodings assuming using associated labels
    parameters:
        encs: TxK*D encoding matrix
        labels: array/list of T labels
        dist_matrix: optional pre-computed (e.g. re-ranked) TxT distance matrix
    """
    if dist_matrix is None:
        dist_matrix = distances(encs)
    # sort each row of the distance matrix
    indices = dist_matrix.argsort()

    n_encs = len(encs)

    mAP = []
    correct = 0
    for r in range(n_encs):
        precisions = []
        rel = 0
        for k in range(n_encs-1):
            if labels[ indices[r,k] ] == labels[ r ]:
                rel += 1
                precisions.append( rel / float(k+1) )
                if k == 0:
                    correct += 1
        avg_precision = np.mean(precisions)
        mAP.append(avg_precision)
    mAP = np.mean(mAP)

    print('Top-1 accuracy: {} - mAP: {}'.format(float(correct) / n_encs, mAP))
    return float(correct) / n_encs, mAP


def rerank_and_evaluate(encs, labels, args, tag=''):
    """
    apply the re-ranking stage on top of the given encodings and evaluate
    """
    if args.rerank == 'none':
        return None

    print('> re-rank{}'.format(tag))
    dist_matrix = rerank_distances(
        encs,
        method=args.rerank,
        qe_k=args.qe_k,
        qe_alpha=args.qe_alpha,
        qe_iter=args.qe_iter,
        k1=args.rerank_k1,
        k2=args.rerank_k2,
        lambda_value=args.rerank_lambda,
    )
    print('> evaluate (re-ranked{})'.format(tag))
    return evaluate(encs, labels, dist_matrix=dist_matrix)


# SIFT + Hellinger normalization (bonus e)

_SIFT = None

def _get_sift():
    global _SIFT
    if _SIFT is None:
        # Works with OpenCV builds that include SIFT
        _SIFT = cv2.SIFT_create()
    return _SIFT

def hellinger_normalize(desc, eps=1e-12):
    """
    Hellinger normalization for SIFT as in the sheet:
    - L1 normalize each descriptor
    - then signed sqrt element-wise
    - NO additional L2 afterwards (sheet explicitly says that)
    """
    if desc is None or len(desc) == 0:
        return np.zeros((0, 128), dtype=np.float32)

    x = desc.astype(np.float32)

    # L1 normalize per row
    l1 = np.sum(np.abs(x), axis=1, keepdims=True) + eps
    x = x / l1

    # signed sqrt
    x = np.sign(x) * np.sqrt(np.abs(x))

    return x.astype(np.float32)

def computeDescs(filename):
    """
    bonus (e):
    - load grayscale image
    - detect SIFT keypoints
    - set all keypoint angles to 0
    - compute SIFT descriptors
    - apply Hellinger normalization
    """
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {filename}")

    sift = _get_sift()

    kps = sift.detect(img, None)
    if kps is None or len(kps) == 0:
        return np.zeros((0, 128), dtype=np.float32)

    # force angle=0 for every keypoint
    for kp in kps:
        kp.angle = 0.0

    kps, desc = sift.compute(img, kps)
    if desc is None:
        return np.zeros((0, 128), dtype=np.float32)

    # Hellinger normalize (L1 + signed sqrt)
    desc = hellinger_normalize(desc)
    return desc

def loadRandomDescriptors_images(files, max_descriptors):
    """
    like loadRandomDescriptors(), but reads IMAGES and computes descriptors on the fly.
    """
    max_files = min(100, len(files))
    indices = np.random.permutation(len(files))[:max_files]
    files_sub = np.array(files)[indices]

    max_descs_per_file = int(max_descriptors / len(files_sub))

    descriptors = []
    for i in tqdm(range(len(files_sub))):
        desc = computeDescs(files_sub[i])
        if desc.shape[0] == 0:
            continue
        take = min(len(desc), max_descs_per_file)
        idx = np.random.choice(len(desc), take, replace=False)
        descriptors.append(desc[idx])

    if len(descriptors) == 0:
        return np.zeros((0, 128), dtype=np.float32)

    descriptors = np.concatenate(descriptors, axis=0)
    return descriptors

def vlad_images(files, mus, powernorm, gmp=False, gamma=1.0):
    """
    like vlad(), but reads IMAGES and computes descriptors on the fly (bonus e).
    supports GMP (bonus f) because it reuses the same logic.
    """
    K = mus.shape[0]
    encodings = []

    for f in tqdm(files):
        desc = computeDescs(f)
        if desc.shape[0] == 0:
            # empty -> return zero vector
            D = mus.shape[1]
            f_enc = np.zeros((D * K), dtype=np.float32)
            encodings.append(f_enc)
            continue

        a = assignments(desc, mus)

        T, D = desc.shape
        f_enc = np.zeros((D * K), dtype=np.float32)

        for k in range(K):
            idx = np.where(a[:, k] == 1)[0]
            if len(idx) == 0:
                continue

            desc_k = desc[idx, :]
            residuals = desc_k - mus[k][None, :]

            if not gmp:
                v_k = residuals.sum(axis=0)
            else:
                # bonus (f): ridge regression with sparse_cg, fit_intercept=False, max_iter=500
                y = np.ones(residuals.shape[0], dtype=np.float32)
                rr = Ridge(alpha=gamma, fit_intercept=False, solver='sparse_cg', max_iter=500)
                rr.fit(residuals, y)
                v_k = rr.coef_.ravel()

            start = k * D
            end = (k + 1) * D
            f_enc[start:end] = v_k.astype(np.float32)

        if powernorm:
            f_enc = np.sign(f_enc) * np.sqrt(np.abs(f_enc))

        f_enc = normalize(f_enc.reshape(1, -1), norm='l2')[0]
        encodings.append(f_enc)

    encodings = np.vstack(encodings)
    return encodings


# multi-VLAD + PCA whitening (bonus g)

def _save_pickle_gz(path, obj):
    with gzip.open(path, 'wb') as fOut:
        cPickle.dump(obj, fOut, -1)

def _load_pickle_gz(path):
    with gzip.open(path, 'rb') as fIn:
        return cPickle.load(fIn)

def compute_codebook(files_train, K, use_images, seed, max_desc=500000, fname=None):
    np.random.seed(seed)

    if fname is not None and os.path.exists(fname):
        return _load_pickle_gz(fname)

    if use_images:
        descriptors = loadRandomDescriptors_images(files_train, max_desc)
    else:
        descriptors = loadRandomDescriptors(files_train, max_desc)

    if descriptors.shape[0] == 0:
        raise RuntimeError("No descriptors found while building the dictionary. Check paths/suffix.")

    mus = dictionary(descriptors, n_clusters=K)

    if fname is not None:
        _save_pickle_gz(fname, mus)

    return mus

def compute_encodings(files, mus, powernorm, gmp, gamma, use_images):
    if use_images:
        return vlad_images(files, mus, powernorm=powernorm, gmp=gmp, gamma=gamma)
    else:
        return vlad(files, mus, powernorm=powernorm, gmp=gmp, gamma=gamma)

def run_multivlad_pca(args, files_train, labels_train, files_test, labels_test):
    K = args.k
    B = args.n_codebooks
    gamma = args.gamma

    # 1) build/load 5 codebooks with different seeds
    mus_list = []
    for i in range(B):
        mus_i_path = f"mus_{K}_cb{i}.pkl.gz"
        mus_i = compute_codebook(
            files_train=files_train,
            K=K,
            use_images=args.use_images,
            seed=42 + i,
            max_desc=500000,
            fname=mus_i_path
        )
        mus_list.append(mus_i)

    # 2) compute VLAD for each codebook, concatenate
    enc_train_list = []
    enc_test_list = []
    for i, mus_i in enumerate(mus_list):
        enc_train_i_path = f"enc_train_K{K}_cb{i}_gmp{int(args.gmp)}_gamma{gamma}_img{int(args.use_images)}.pkl.gz"
        enc_test_i_path  = f"enc_test_K{K}_cb{i}_gmp{int(args.gmp)}_gamma{gamma}_img{int(args.use_images)}.pkl.gz"

        if os.path.exists(enc_train_i_path) and not args.overwrite:
            enc_train_i = _load_pickle_gz(enc_train_i_path)
        else:
            enc_train_i = compute_encodings(files_train, mus_i, args.powernorm, args.gmp, gamma, args.use_images)
            _save_pickle_gz(enc_train_i_path, enc_train_i)

        if os.path.exists(enc_test_i_path) and not args.overwrite:
            enc_test_i = _load_pickle_gz(enc_test_i_path)
        else:
            enc_test_i = compute_encodings(files_test, mus_i, args.powernorm, args.gmp, gamma, args.use_images)
            _save_pickle_gz(enc_test_i_path, enc_test_i)

        enc_train_list.append(enc_train_i)
        enc_test_list.append(enc_test_i)

    enc_train_multi = np.concatenate(enc_train_list, axis=1)
    enc_test_multi = np.concatenate(enc_test_list, axis=1)

    # 3) PCA whitening to pca_dim on TRAIN (or subset)
    pca_path = f"pca_K{K}_B{B}_dim{args.pca_dim}_img{int(args.use_images)}_gmp{int(args.gmp)}_gamma{gamma}.pkl.gz"
    if os.path.exists(pca_path) and not args.overwrite:
        pca = _load_pickle_gz(pca_path)
    else:
        n_fit = min(args.pca_subset, enc_train_multi.shape[0])
        idx = np.random.permutation(enc_train_multi.shape[0])[:n_fit]
        X_fit = enc_train_multi[idx]

        pca = PCA(n_components=args.pca_dim, whiten=True, random_state=42)
        pca.fit(X_fit)
        _save_pickle_gz(pca_path, pca)

    enc_train_pca = pca.transform(enc_train_multi).astype(np.float32)
    enc_test_pca = pca.transform(enc_test_multi).astype(np.float32)

    # l2 normalize after PCA (recommended for cosine distance)
    enc_train_pca = normalize(enc_train_pca, norm='l2', axis=1)
    enc_test_pca = normalize(enc_test_pca, norm='l2', axis=1)

    print('> evaluate multi-VLAD + PCA')
    evaluate(enc_test_pca, labels_test)
    rerank_and_evaluate(enc_test_pca, labels_test, args, tag=' multi-VLAD + PCA')

    print('> evaluate multi-VLAD + PCA + E-SVM')
    enc_test_esvm = esvm(enc_test_pca, enc_train_pca, C=args.C)
    evaluate(enc_test_esvm, labels_test)
    rerank_and_evaluate(enc_test_esvm, labels_test, args, tag=' multi-VLAD + PCA + E-SVM')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('retrieval')
    parser = parseArgs(parser)
    args = parser.parse_args()
    if args.use_images:
        loadRandomDescriptors = loadRandomDescriptors_images
        vlad = vlad_images
    if args.multivlad:
        files_train, labels_train = getFiles(args.in_train, args.suffix, args.labels_train)
        files_test, labels_test = getFiles(args.in_test, args.suffix, args.labels_test)

        print('#train: {}'.format(len(files_train)))
        print('#test: {}'.format(len(files_test)))

        run_multivlad_pca(args, files_train, labels_train, files_test, labels_test)
        sys.exit(0)

    np.random.seed(42) # fixed random seed
   
    # a) dictionary
    files_train, labels_train = getFiles(args.in_train, args.suffix,
                                         args.labels_train)
    print('#train: {}'.format(len(files_train)))
    mus_fname = 'mus_images.pkl.gz' if args.use_images else 'mus.pkl.gz'
    if not os.path.exists(mus_fname):
        
        descriptors = loadRandomDescriptors(files_train, max_descriptors=500000)
        print('> loaded {} descriptors:'.format(len(descriptors)))

        # cluster centers
        print('> compute dictionary')
        
        mus = dictionary(descriptors, n_clusters=100)
        with gzip.open(mus_fname, 'wb') as fOut:
            cPickle.dump(mus, fOut, -1)
    else:
        with gzip.open(mus_fname, 'rb') as f:
            mus = cPickle.load(f, encoding='latin1')

  
    # b) VLAD encoding
    print('> compute VLAD for test')
    files_test, labels_test = getFiles(args.in_test, args.suffix,
                                       args.labels_test)
    print('#test: {}'.format(len(files_test)))
    gamma = args.gamma
    cache_suffix = '_images' if args.use_images else ''
    fname = ('enc_test{}_gmp{}.pkl.gz'.format(cache_suffix, gamma)
             if args.gmp else 'enc_test{}.pkl.gz'.format(cache_suffix))
    if not os.path.exists(fname) or args.overwrite:
        enc_test = vlad(files_test, mus, powernorm=args.powernorm,
                gmp=args.gmp, gamma=args.gamma)
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_test, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_test = cPickle.load(f)
   
    # cross-evaluate test encodings
    print('> evaluate')
    evaluate(enc_test, labels_test)

    # c) re-ranking on the plain VLAD encodings
    rerank_and_evaluate(enc_test, labels_test, args)

    # d) compute exemplar svms
    print('> compute VLAD for train (for E-SVM)')
    fname = ('enc_train{}_gmp{}.pkl.gz'.format(cache_suffix, gamma)
             if args.gmp else 'enc_train{}.pkl.gz'.format(cache_suffix))
    if not os.path.exists(fname) or args.overwrite:
        enc_train = vlad(files_train, mus, powernorm=args.powernorm,
                     gmp=args.gmp, gamma=args.gamma)
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_train, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_train = cPickle.load(f)

    print('> esvm computation')
    enc_test = esvm(enc_test, enc_train, C=args.C)

    # eval
    print('> evaluate (E-SVM)')
    evaluate(enc_test, labels_test)

    # e) re-ranking on top of the E-SVM embedding
    rerank_and_evaluate(enc_test, labels_test, args, tag=' + E-SVM')
