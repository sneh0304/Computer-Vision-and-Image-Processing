import sys
import os
import json
import pickle
import numpy as np
import cv2

# method to calculate haar features based on size of the image
def get_haar_features(width, height):
    '''
        a b c d
        e f g h
        i j k
        l m
    '''
    haar_features = []
    for W in range(1, width + 1):
        for H in range(1, height + 1):
            for x in range(width - W):
                for y in range(height - H):
                    a = (x, y)
                    b = (x + W, y)
                    c = (x + 2 * W, y)
                    d = (x + 3 * W, y)
                    e = (x, y + H)
                    f = (x + W, y + H)
                    g = (x + 2 * W, y + H)
                    h = (x + 3 * W, y + H)
                    i = (x, y + 2 * H)
                    j = (x + W, y + 2 * H)
                    k = (x + 2 * W, y + 2 * H)
                    l = (x, y + 3 * H)
                    m = (x + W, y + 3 * H)

                    '''
                        a b c
                        e f g

                        A=a B=b C=c
                        D=e E=f F=g
                    '''
                    if x + 2 * W < width:
                        haar_features.append(haar_feature(feature_type = 1, a = a, b = b, c = c, d = e, e = f, f = g))

                    '''
                        a b
                        e f
                        i j

                        A=a D=b
                        B=e E=f
                        C=i F=j
                    '''
                    if y + 2 * H < height:
                        haar_features.append(haar_feature(feature_type = 2, a = a, b = e, c = i, d = b, e = f, f = j))

                    '''
                        a b c d
                        e f g h

                        A=a B=b C=c D=d
                        E=e F=f G=g H=h
                    '''
                    if x + 3 * W < width:
                        haar_features.append(haar_feature(feature_type = 3, a = a, b = b, c = c, d = d, e = e, f = f, g = g, h = h))

                    '''
                        a b
                        e f
                        i j
                        l m

                        A=a E=b
                        B=e F=f
                        C=i G=j
                        D=l H=m
                    '''
                    if y + 3 * H < height:
                        haar_features.append(haar_feature(feature_type = 4, a = a, b = e, c = i, d = l, e = b, f = f, g = j, h = m))

                    '''
                        a b c
                        e f g
                        i j k

                        A=a B=b C=c
                        D=e E=f F=g
                        G=i H=j I=k
                    '''

                    if x + 2 * W < width and y + 2 * H < height:
                        haar_features.append(haar_feature(feature_type = 5, a = a, b = b, c = c, d = e, e = f, f = g, g = i, h = j, i = k))
    return haar_features

# class for haar feature, which contains variables for feature type and co-ordinates for point A-I
# and a method to compute the value of that particular feature in a given image
class haar_feature:
    def __init__(self, feature_type, a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 0, i = 0):
        self.Type = feature_type
        self.A = a
        self.B = b
        self.C = c
        self.D = d
        self.E = e
        self.F = f
        self.G = g
        self.H = h
        self.I = i
    def compute(self, img):
        res = 0
        if self.Type == 1 or self.Type == 2:
            '''
                Type 1:
                    A B C
                    D E F
                res1 = D + F + 2B - A - C - 2E

                Type 2:
                    A D
                    B E
                    C F
                res2 = A + C + 2E - D - F - 2B
                => res2 = -res1
            '''
            res = img[self.D[1]][self.D[0]] + img[self.F[1]][self.F[0]] + (2 * img[self.B[1]][self.B[0]]) - img[self.A[1]][self.A[0]] - img[self.C[1]][self.C[0]] - (2 * img[self.E[1]][self.E[0]])
            return res if self.Type == 1 else -res
        elif self.Type == 3 or self.Type == 4:
            '''
                Type 3:
                    A B C D
                    E F G H
                res3 = D + E + 2B + 2G - A - H - 2C - 2F

                Type 4:
                    A E
                    B F
                    C G
                    D H
                res4 = D + E + 2B + 2G - A - H - 2C - 2F
                => res4 = res3
            '''
            res = img[self.D[1]][self.D[0]] + img[self.E[1]][self.E[0]] + (2 * img[self.B[1]][self.B[0]]) + (2 * img[self.G[1]][self.G[0]]) - img[self.A[1]][self.A[0]] - img[self.H[1]][self.H[0]] - (2 * img[self.C[1]][self.C[0]]) - (2 * img[self.F[1]][self.F[0]])
            return res
        else:
            '''
                Type 1:
                    A B C
                    D E F
                    G H I
                res5 = 2B + 2D + 2F + 2H - A - C - G - I - 4E
            '''
            res = (2 * img[self.B[1]][self.B[0]]) + (2 * img[self.D[1]][self.D[0]]) + (2 * img[self.F[1]][self.F[0]]) + (2 * img[self.H[1]][self.H[0]]) - img[self.A[1]][self.A[0]] - img[self.C[1]][self.C[0]] - img[self.G[1]][self.G[0]] - img[self.I[1]][self.I[0]] - (4 * img[self.E[1]][self.E[0]])
            return res

# method to compute all the feature values of a given image
def calculate_feature_values(features, img):
    feature_values = []
    for feature in features:
        feature_values.append(feature.compute(img))
    return np.array(feature_values)

# Ref: https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf
# class for a weak classifier which contains variables for a feature, a threshold and a parity
# and a classify method to classify as 0 or 1 as per viola jones paper
class weak_classifier:
    def __init__(self, feature, threshold, parity):
        self.feature = feature
        self.threshold = threshold
        self.parity = parity

    def classify(self, integral_img):
        val = self.feature.compute(integral_img)
        return 1 if self.parity * val < self.parity * self.threshold else 0

# Ref: https://medium.com/datadriveninvestor/understanding-and-implementing-the-viola-jones-image-classification-algorithm-85621f7fe20b
# Ref: https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf
# class for the main viola jones algorithm as per viola jones paper
class viola_jones():
    def __init__(self, n):
        self.N = n
        self.Classifiers = list()
        self.Alphas = list()

    def train(self, training_imgs, train_y, haar_features):
        face_len = len(train_y[train_y == 1])
        non_face_len = len(train_y[train_y == 0])
        train_x = list()
        weights = np.zeros(len(training_imgs))

        for i, img in enumerate(training_imgs):
            if train_y[i] == 1:
                weights[i] = 1.0 / (2 * face_len)
            else:
                weights[i] = 1.0 / (2 * non_face_len)
        with open('feature_values.pkl', 'rb') as f:
            features_values = pickle.load(f)
        print('feature values loaded')
        train_x = np.array(features_values, dtype = int32)

        # trainig the whole 63,960 features will take a lot of time, hence we are selecting the top 15% of
        # the features out of 63,960 based on f1 score. This is done to improve the performance of training.
        selection_model = sk.SelectPercentile(sk.f_classif, percentile = 10)
        selection_model.fit(train_x, train_y)
        selected_idx = selection_model.get_support(indices = True)
        train_x = train_x[:, selected_idx]
        haar_features = haar_features[selected_idx]

        for n in range(self.N):
            weights = weights / np.sum(weights)
            weak_classifiers = self.train_wc(train_x, train_y, haar_features, weights)
            classifier, error, errorList = self.select_best_classifier(weak_classifiers, weights, training_imgs, train_y)
            beta = error / (1.0 - error)
            weights = np.multiply(weights, np.power(beta, np.subtract(1, errorList)))
            alpha = np.log(1.0 / beta)
            self.Alphas.append(alpha)
            self.Classifiers.append(classifier)

    def train_wc(self, train_x, train_y, features, weights):
        total_pos_weights = np.sum(weights[train_y == 1])
        total_neg_weights = np.sum(weights[train_y == 0])

        classifiers = list()

        for i, feature_values in enumerate(train_x.T):
            sorted_idx = np.argsort(feature_values)
            sorted_feature_values = feature_values[sorted_idx]
            sorted_weights = weights[sorted_idx]
            sorted_y = train_y[sorted_idx]
            positives = negatives = 0
            pos_weights_so_far = neg_weights_so_far = 0
            min_error, best_feature, best_theta, best_parity = float('inf'), None, None, None

            for w, f, y in zip(sorted_weights, sorted_feature_values, sorted_y):
                error = min(total_pos_weights + neg_weights_so_far - pos_weights_so_far, total_neg_weights + pos_weights_so_far - neg_weights_so_far)
                if error < min_error:
                    min_error = error
                    best_feature = features[i]
                    best_theta = f
                    best_parity = 1 if positives > negatives else -1
                if y == 1:
                    positives += 1
                    pos_weights_so_far += w
                else:
                    negatives += 1
                    neg_weights_so_far += w

            classifier = weak_classifier(best_feature, best_theta, best_parity)
            classifiers.append(classifier)

        return classifiers

    def select_best_classifier(self, weak_classifiers, weights, training_imgs, train_y):
        best_classifier, best_error, best_acc = None, float('inf'), None
        for classifier in weak_classifiers:
            errorSum = 0
            errorList = list()
            for img, y, w in zip(training_imgs, train_y, weights):
                error = abs(classifier.classify(img) - y)
                errorList.append(error)
                errorSum += w * error
            errorSum = errorSum / len(training_imgs)

            if errorSum < best_error:
                best_classifier = classifier
                best_error = errorSum
                best_errorList = errorList
        return best_classifier, best_error, best_errorList

    def classify(self, img):
        res = 0
        for alpha, classifier in zip(self.Alphas, self.Classifiers):
            res += alpha * classifier.classify(img)

        return 1 if res >= 0.5 * np.sum(self.Alphas) else 0

# class for cascade classifier which calls viola_jones class
class cascade_classifier:
    def __init__(self, layers):
        self.Layers = layers
        self.VJ = list()
        self.Features = np.array(get_haar_features(20, 20))

    def train(self, x, y):
        for n in self.Layers:
            vj = viola_jones(n)
            vj.train(x, y, self.Features)
            self.VJ.append(vj)
            print('Training done for %d layer' %(n))

    def classify(self, img):
        for vj in self.VJ:
            if vj.classify(img) == 0:
                return 0
        return 1

    def save(self):
        cascade_path = os.path.join('Model_files', 'cascade_classifier.pkl')
        with open(cascade_path, 'wb') as f:
            pickle.dump(self, f)

# method to compute the fearure values of all the images and train the cascade classifier
def trainModel():
    trainData_path = os.path.join('Model_files', 'train_data.pkl')
    with open(trainData_path, 'rb') as f:
        train = pickle.load(f)

    features = get_haar_features(20, 20)
    train = np.array(train)
    x = np.array([i for i in train[:, 0]])
    y = train[:, 1]
    x = integral_image(x)
    features_values = []
    for img in x:
        features_values.append(calculate_feature_values(features, img))

    featureValues_path = os.path.join('Model_files', 'feature_values.pkl')
    with open(featureValues_path, 'wb') as f:
        pickle.dump(features_values, f)

    layers = [1, 10, 25, 25, 50]
    cc = cascade_classifier(layers)
    cc.train(x, y)
    cc.save()

# method to compute integral image
def integral_image(img):
    integral_img = img.copy()
    integral_img = np.cumsum(integral_img, axis = -1)
    integral_img = np.cumsum(integral_img, axis = -2)
    return integral_img

# method to detect faces in an image and return all the possible hits in a list
def detect(image):
    res = []
    height, width = image.shape
    window_size = 20
    scaling_factor = 1.5
    while window_size < height and window_size < width:
        for y in range(0, height - window_size, 20):
            for x in range(0, width - window_size, 20):
                img = cv2.resize(image[y : y + window_size, x : x + window_size], (20, 20))
                img = integral_image(img)
                if cascade_classifier.classify(img) == 1:
                    res.append([x, y, window_size, window_size])
        window_size = int(window_size * scaling_factor)

    return res

def main(train = False):
    if train:
        trainModel()

    global cascade_classifier
    classifier_path = os.path.join('Model_files', 'cascade_classifier.pkl')
    with open(classifier_path, 'rb') as f:
        cascade_classifier = pickle.load(f)

    folder = sys.argv[1]

    final_result = []
    img_list = os.listdir(folder)
    for i, img_name in enumerate(img_list):
        if '.jpg' in img_name:
            img = cv2.imread(os.path.join(folder, img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected_faces = detect(img)
            groupRectangles = cv2.groupRectangles(detected_faces, 2, 0.5)
            for bbox in groupRectangles[0]:
                face = dict()
                face['iname'] = img_name
                face['bbox'] = [int(j) for j in bbox]
                final_result.append(face)
            if (i + 1) % 20 == 0:
                print('done for %d images' %(i + 1))

    out_path = os.path.join(folder, 'results.json')
    with open(out_path, 'w') as f:
        json.dump(final_result, f)


if __name__ == '__main__':
    main()
