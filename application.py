import tensorflow
from keras import datasets, layers, models
import numpy as np

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# 2 boyutlu diziyi tek boyutlu hale dönüştürüyoruz
y_test = y_test.reshape(
    -1,
)

image_classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# normalizasyon | doğru sonuç almak için renk pixeline bölüyoruz
X_train = X_train / 255
X_test = X_test / 255

deep_learning_model = models.Sequential(
    [
        layers.Conv2D(  # bu görsellerden belirli özellikleri yakalamaya çalışır.
            filters=32, kernel_size=(3, 3), activation="relu", input_shape=(32, 32, 3)
        ),
        layers.MaxPooling2D(
            (2, 2)
        ),  # en büyük pixel değerini alır en belirgin özellikleri ortaya çıkarır işlem hızını etiler
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),  # CNN ve ANN i otomatik olarak bağlıyoruz
        layers.Dense(
            64, activation="relu"
        ),  # relu negatif değerler için 0 pozitif x değerler için x i alır. böylece ağ daha hızlı eğitilir
        layers.Dense(
            10, activation="softmax"
        ),
        # softmax sınıflandırma problemlerinde kullanılır her girdinin bir sınıfa ait olmasını gösteren 0 veya 1 degerini alır
    ]
)

deep_learning_model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

deep_learning_model.fit(X_train, y_train, epochs=5)

deep_learning_model.evaluate(X_test, y_test)  # model başarısını ölçüyoruz

prediction = deep_learning_model.predict(X_test)

predictions_classes = [np.argmax(element) for element in prediction]

for i in range(10):
    test_result = image_classes[y_test[i]]
    prediction_result = image_classes[predictions_classes[i]]
    if str(test_result) == str(prediction_result):
        print("test verisi: " + str(test_result) + " Tahmin: " + str(prediction_result) + " || DOĞRU")
    else:
        print("test verisi: " + str(test_result) + " Tahmin: " + str(prediction_result) + " || YANLIŞ")
