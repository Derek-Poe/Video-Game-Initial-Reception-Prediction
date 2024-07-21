import re as regex
import nltk as nltk
import langdetect as ldetect
import langid as lId
import psutil as psutil
import dask.dataframe as daskdf
import dask.distributed as daskdist
import dask.cache as DaskCache
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.preprocessing as tfkpp
import tensorflow.keras.models as tfkm
import tensorflow.keras.layers as tfkl
import tensorflow.keras.callbacks as tfkc
import gc as gc
import io as io
import sklearn.preprocessing as skpp
import sklearn.model_selection as skms
import sklearn.metrics as skm
import collections as colls
import pickle as pkl

nltk.download("stopwords")

def GetUniqueLabels():
    mcData = daskdf.read_csv(mcPath, low_memory = False, dtype = mcDaskDT)
    mcData = mcData[["Genre", "Platforms", "Game Developer", "Product Rating"]].dropna()
    mcData["Genre"] = mcData["Genre"].str[9:]
    devCounter = colls.Counter(list(mcData["Game Developer"].str.split(",").explode().compute()))
    devFilter = {dev for dev, c in devCounter.items() if c > 100}
    return {
        "Genre": list(mcData["Genre"].str.split(",").explode().unique().compute()),
        "Platforms": list(mcData["Platforms"].str.split(",").explode().unique().compute()),
        "Game Developer": list(devFilter),
        "Product Rating": list(mcData["Product Rating"].unique().compute())
    }

def GetTokenVars():
    mcData = daskdf.read_csv(mcPath, low_memory = False, dtype = mcDaskDT)
    mcData = mcData["Review Text"].dropna()
    def GetReviewLen(rev):
        if pd.isna(rev) or not rev.strip():
            return 0, []
        rWords = regex.findall(r"\b\w+\b", rev.lower())
        rWords = [word for word in rWords if word not in langPro["nltkSW"]]
        if not rWords:
            return 0, []
        try:
            jStr = (" ".join(rWords))
            langCheck1 = langPro["ldect"](jStr)
            langCheck2 = langPro["lId"](jStr)[0]
            if langCheck1 != "en" and langCheck2 != "en":
                return 0, []
        except:
            return 0, []
        return len(rWords), rWords
    lensAndWords = mcData.apply(GetReviewLen, meta = pd.Series(dtype = object)).compute()
    maxSeqLen = max(lensAndWords.map(lambda x: x[0]))
    allWords = [word for words in lensAndWords.map(lambda x: x[1]) for word in words]
    allWords = list(set(allWords))
    return maxSeqLen, allWords

def ReviewTextPrep(rTxt, langPro, mcTokenizer, maxSeqLen):
    try:
        langCheck1 = langPro["ldect"](rTxt)
        langCheck2 = langPro["lId"](rTxt)[0]
        if langCheck1 == "en" or langCheck2 == "en":
            rTxt = rTxt.lower()
            rTxt = regex.sub(r"[^\w\s]", "", rTxt)
            rTxt = regex.sub(r"\d+", "", rTxt)
            rTxt = rTxt.split()
            rTxt = [word for word in rTxt if word not in langPro["nltkSW"]]
            if not rTxt:
                return "~NE~"
            rTxt = mcTokenizer.texts_to_sequences([rTxt])
            rTxt = tfkpp.sequence.pad_sequences(rTxt, maxlen = maxSeqLen, padding = "post")
            rTxt = np.clip(rTxt, 0, mcTokenizer.num_words - 1)
            return rTxt[0]
        else:
            return "~NE~"
    except:
        return "~NE~"

def PreprocessData(mcDataChunk, uniqueLabels):
    allVars = ["Game Release Date", "Game Developer", "Genre", "Platforms", "Product Rating", "Overall Metascore", "Overall User Rating", "Rating Given By The Reviewer", "Review Text"]
    mcDataChunk = mcDataChunk[allVars]
    mcDataChunk = mcDataChunk.drop_duplicates()
    mcDataChunk = mcDataChunk.dropna(subset = ["Review Text", "Overall User Rating", "Overall Metascore", "Game Developer", "Rating Given By The Reviewer"])
    mcDataChunk["Product Rating"] = mcDataChunk["Product Rating"].fillna(mcDataChunk["Product Rating"].mode().compute()[0])
    mcDataChunk = mcDataChunk[mcDataChunk["Overall User Rating"] != "tbd"]
    mcDataChunk["Overall User Rating"] = mcDataChunk["Overall User Rating"].astype(np.float32)
    mcDataChunk["Game Release Date"] = daskdf.to_datetime(mcDataChunk["Game Release Date"], format = "%d-%b-%y", errors = "coerce")
    mcDataChunk = mcDataChunk.dropna(subset = ["Game Release Date"])
    mcDataChunk["Release Year"] = mcDataChunk["Game Release Date"].dt.year
    mcDataChunk["Release Quarter"] = mcDataChunk["Game Release Date"].dt.quarter
    mcDataChunk["Release Month"] = mcDataChunk["Game Release Date"].dt.month
    mcDataChunk["Release Week"] = np.floor(mcDataChunk["Game Release Date"].dt.day / 10).astype(int)
    mcDataChunk["Release DayOfWeek"] = mcDataChunk["Game Release Date"].dt.day_of_week
    mcDataChunk["Release DayOfYear"] = mcDataChunk["Game Release Date"].dt.day_of_year
    mcDataChunk = mcDataChunk.drop(["Game Release Date"], axis = 1)
    mcDataChunk["Genre"] = mcDataChunk["Genre"].str[9:]
    mcDataChunk["Genre"] = mcDataChunk["Genre"].str.split(",")
    mcDataChunk["Platforms"] = mcDataChunk["Platforms"].str.split(",")
    mcDataChunk["Game Developer"] = mcDataChunk["Game Developer"].str.split(",")
    mcDataChunk["Product Rating"] = mcDataChunk["Product Rating"].str.split(",")
    mlbGenre = skpp.MultiLabelBinarizer(classes = uniqueLabels["Genre"])
    mlbPlatforms = skpp.MultiLabelBinarizer(classes = uniqueLabels["Platforms"])
    mlbDeveloper = skpp.MultiLabelBinarizer(classes = uniqueLabels["Game Developer"])
    mlbRating = skpp.MultiLabelBinarizer(classes = uniqueLabels["Product Rating"])
    mlbGenreData = mlbGenre.fit_transform(mcDataChunk["Genre"].compute())
    mlbPlatformsData = mlbPlatforms.fit_transform(mcDataChunk["Platforms"].compute())
    mlbDeveloperData = mlbDeveloper.fit_transform(mcDataChunk["Game Developer"].compute())
    mlbRatingData = mlbRating.fit_transform(mcDataChunk["Product Rating"].compute())
    genrePDF = pd.DataFrame(mlbGenreData, columns = mlbGenre.classes_, index = mcDataChunk.index.compute())
    platformsPDF = pd.DataFrame(mlbPlatformsData, columns = mlbPlatforms.classes_, index = mcDataChunk.index.compute())
    developerPDF = pd.DataFrame(mlbDeveloperData, columns = mlbDeveloper.classes_, index = mcDataChunk.index.compute())
    ratingPDF = pd.DataFrame(mlbRatingData, columns = mlbRating.classes_, index = mcDataChunk.index.compute())
    mcDataChunk = mcDataChunk.drop(["Genre", "Platforms", "Game Developer", "Product Rating"], axis = 1)
    mcDataChunk = mcDataChunk.join(daskdf.from_pandas(genrePDF, npartitions = 1))
    mcDataChunk = mcDataChunk.join(daskdf.from_pandas(platformsPDF, npartitions = 1))
    mcDataChunk = mcDataChunk.join(daskdf.from_pandas(developerPDF, npartitions = 1))
    mcDataChunk = mcDataChunk.join(daskdf.from_pandas(ratingPDF, npartitions = 1))
    daskdist.wait(mcDataChunk)
    del mlbGenre, mlbPlatforms, mlbDeveloper, mlbRating, mlbGenreData, mlbPlatformsData, mlbDeveloperData, mlbRatingData, genrePDF, platformsPDF, developerPDF, ratingPDF
    gc.collect()
    return mcDataChunk

def FinalPreprocessAndTrain(mcDataChunk, langPro, mcMod, mcTokenizer, maxSeqLen, mcScaler):
    mcDataChunk["Review Text"] = mcDataChunk["Review Text"].map_partitions(lambda mcDF: mcDF.apply(ReviewTextPrep, langPro = langPro, mcTokenizer = mcTokenizer, maxSeqLen = maxSeqLen), meta = pd.Series(dtype = "object", name = "Review Text"))
    mcDataChunk = mcDataChunk[mcDataChunk["Review Text"].apply(lambda x: "~NE~" not in x, meta = ("Review Text", "bool"))]
    mcDataChunk = mcDataChunk.compute()
    mcDataChunk = mcDataChunk.reset_index(drop = True)
    mcSeqs = np.stack(mcDataChunk["Review Text"].values)
    mcDataChunk = mcDataChunk.drop(["Review Text"], axis = 1)
    mcDataChunk = mcDataChunk.join(pd.DataFrame(mcSeqs, index = mcDataChunk.index))
    mcY = mcDataChunk["Overall Metascore"].copy().astype(np.float32)
    mcX = mcDataChunk.copy().drop(["Overall Metascore"], axis = 1).astype(np.float32)
    del mcSeqs, mcDataChunk
    gc.collect()
    mcX[numVars] = mcScaler.transform(mcX[numVars])
    mcXTrain, mcXTest, mcYTrain, mcYTest = skms.train_test_split(mcX, mcY, test_size = 0.2, random_state = 17)
    numVars = ["Overall User Rating", "Release Year", "Release Quarter", "Release Month", "Release Week", "Release DayOfWeek", "Release DayOfYear"]
    mcXTrain = mcXTrain.to_numpy()
    mcXTest = mcXTest.to_numpy()
    mcTFDataTrain = tf.data.Dataset.from_tensor_slices(((mcXTrain[:, -maxSeqLen:], mcXTrain[:, :-maxSeqLen]), mcYTrain.values)).batch(32)
    mcTFDataTest = tf.data.Dataset.from_tensor_slices(((mcXTest[:, -maxSeqLen:], mcXTest[:, :-maxSeqLen]), mcYTest.values)).batch(32)
    mcMod.fit(mcTFDataTrain, validation_data = mcTFDataTest, epochs = 20, callbacks = [tfkc.EarlyStopping(monitor = "val_loss", patience = 5, restore_best_weights = True)])
    return mcMod

def SaveVar(var, fName):
    with open(f"intermediate/{fName}.pkl", "wb") as wb:
        pkl.dump(var, wb)

def LoadVar(fName):
    with open(f"intermediate/{fName}.pkl", "rb") as rb:
        return pkl.load(rb)
    
def SaveModVars():
    SaveVar(mcScaler, "../mcScaler")
    SaveVar(mcTokenizer, "../mcTokenizer")
    SaveVar(maxSeqLen, "../maxSeqLen")
    SaveVar(uniqueLabels, "../uniqueLabels")

def LoadModVars():
    with open("mcScaler.pkl", "rb") as rb:
        mcScaler = pkl.load(rb)
    with open("mcTokenizer.pkl", "rb") as rb:
        mcTokenizer = pkl.load(rb)
    with open("maxSeqLen.pkl", "rb") as rb:
        maxSeqLen = pkl.load(rb)
    with open("uniqueLabels.pkl", "rb") as rb:
        uniqueLabels = pkl.load(rb)
    return mcScaler, mcTokenizer, maxSeqLen, uniqueLabels

def EvaluateMCModel(mcMod, mcScaler, mcTokenizer, maxSeqLen, uniqueLabels):
    mcDataChunk = daskdf.read_csv(mcPath, low_memory = False, dtype = mcDaskDT).sample(frac = 0.01)
    mcDataChunk = PreprocessData(mcDataChunk, uniqueLabels)
    mcDataChunk["Review Text"] = mcDataChunk["Review Text"].map_partitions(lambda mcDF: mcDF.apply(ReviewTextPrep, langPro = langPro, mcTokenizer = mcTokenizer, maxSeqLen = maxSeqLen), meta = pd.Series(dtype = "object", name = "Review Text"))
    mcDataChunk = mcDataChunk[mcDataChunk["Review Text"].apply(lambda x: "~NE~" not in x, meta = ("Review Text", "bool"))]
    mcDataChunk = mcDataChunk.compute()
    mcDataChunk = mcDataChunk.reset_index(drop = True)
    mcSeqs = np.stack(mcDataChunk["Review Text"].values)
    mcDataChunk = mcDataChunk.drop(["Review Text"], axis = 1)
    mcDataChunk = mcDataChunk.join(pd.DataFrame(mcSeqs, index = mcDataChunk.index))
    numVars = ["Overall User Rating", "Release Year", "Release Quarter", "Release Month", "Release Week", "Release DayOfWeek", "Release DayOfYear"]
    mcDataChunk[numVars] = mcScaler.transform(mcDataChunk[numVars])
    mcY = mcDataChunk["Overall Metascore"].copy().astype(np.float32).to_numpy()
    mcX = mcDataChunk.copy().drop(["Overall Metascore"], axis = 1).astype(np.float32).to_numpy()
    del mcSeqs, mcDataChunk
    gc.collect()
    mcTestText = mcX[:, -maxSeqLen:]
    mcTestNum = mcX[:, :-maxSeqLen]
    mcTestPreds = mcMod.predict([mcTestText, mcTestNum])
    mcMSE = skm.mean_squared_error(mcY, mcTestPreds)
    mcMAE = skm.mean_absolute_error(mcY, mcTestPreds)
    mcR2 = skm.r2_score(mcY, mcTestPreds)
    return mcMSE, mcMAE, mcR2

def PredictMetascore(mcMod, mcTokenizer, mcScaler, maxSeqLen, inData):
    inData = pd.DataFrame([inData])
    inData = PreprocessData(daskdf.from_pandas(inData, npartitions = 1), uniqueLabels)
    inData["Review Text"] = inData["Review Text"].map_partitions(lambda mcDF: mcDF.apply(ReviewTextPrep, langPro = langPro, mcTokenizer = mcTokenizer, maxSeqLen = maxSeqLen), meta = pd.Series(dtype = "object", name = "Review Text"))
    inData = inData[inData["Review Text"].apply(lambda x: "~NE~" not in x, meta = ("Review Text", "bool"))]
    inData = inData.compute()
    inData = inData.reset_index(drop = True)
    mcSeqs = np.stack(inData["Review Text"].values)
    inData = inData.drop(["Review Text"], axis = 1)
    inData = inData.join(pd.DataFrame(mcSeqs, index = inData.index))
    numVars = ["Overall User Rating", "Release Year", "Release Quarter", "Release Month", "Release Week", "Release DayOfWeek", "Release DayOfYear"]
    inData[numVars] = mcScaler.transform(inData[numVars])
    inX = inData.copy().drop(["Overall Metascore"], axis = 1).astype(np.float32).to_numpy()
    del mcSeqs, inData
    gc.collect()
    inText = inX[:, -maxSeqLen:]
    inNum = inX[:, :-maxSeqLen]
    mcPred = mcMod.predict([inText, inNum])
    return mcPred

ldetect.DetectorFactory.seed = 17
langPro = {
    "ldect": ldetect.detect,
    "lId": lId.langid.classify,
    "nltkSW": set(nltk.corpus.stopwords.words("english"))
}
mcDaskDT = {
    "Review Date": "object",
    "Overall User Rating": "object"
}

mcPath = "metacritic_pc_games.csv"
rootChunkSize = 10000
with io.open(mcPath, "r", encoding = "utf-8", buffering = 1024**3) as mcFile:
    mcHeaders = mcFile.readline().strip().split(",")
    mcEntryCount = sum(1 for line in mcFile) - 1

chunkCount = psutil.cpu_count()
memUtil = 0.75
daskCache = DaskCache.Cache(2e9)
daskCache.register()
daskClient = daskdist.Client(n_workers = 2, threads_per_worker = 4, memory_limit = f"{int(psutil.virtual_memory().total / (1024 ** 2) * memUtil / 2)}MB")

maxSeqLen, allWords = GetTokenVars()
SaveVar(maxSeqLen, "maxSeqLen")
SaveVar(allWords, "allWords")
# maxSeqLen = LoadVar("maxSeqLen")
# allWords = LoadVar("allWords")

daskClient.shutdown()
daskClient = daskdist.Client(n_workers = chunkCount, threads_per_worker = 2, memory_limit = f"{int(psutil.virtual_memory().total / (1024 ** 2) * memUtil / chunkCount)}MB")

uniqueLabels = GetUniqueLabels()
SaveVar(uniqueLabels, "uniqueLabels")
# uniqueLabels = LoadVar("uniqueLabels")

initChunk = daskdf.read_csv(mcPath, low_memory = False, dtype = mcDaskDT)
initChunk = PreprocessData(initChunk.sample(frac = 0.01), uniqueLabels)
numVars = ["Overall User Rating", "Release Year", "Release Quarter", "Release Month", "Release Week", "Release DayOfWeek", "Release DayOfYear"]
mcScaler = skpp.StandardScaler().fit(initChunk[numVars].compute())
SaveVar(mcScaler, "mcScaler")

mcTokenizer = tfkpp.text.Tokenizer(num_words = len(allWords) + 1)
mcTokenizer.fit_on_texts(allWords)
SaveVar(mcTokenizer, "mcTokenizer")
vocabSize = len(mcTokenizer.word_index) + 1

numShape = len(numVars) + initChunk.drop(["Review Text", "Overall Metascore"] + numVars, axis = 1).shape[1]
inShape = maxSeqLen + numShape
#text branch
textInLayer = tfkl.Input(shape = (maxSeqLen,), name = "textInLayer")
embedLayer = tfkl.Embedding(input_dim = vocabSize, output_dim = 128)(textInLayer)
lstmLayer = tfkl.LSTM(64)(embedLayer)
dropoutLayer = tfkl.Dropout(0.5)(lstmLayer)
#numeric branch
numInLayer = tfkl.Input(shape = (numShape,), name = "numInLayer")
dense1Layer = tfkl.Dense(64, activation = "relu")(numInLayer)
dropout1Layer = tfkl.Dropout(0.5)(dense1Layer)
#concatenated output
outLayerConcat = tfkl.concatenate([dropoutLayer, dropout1Layer])
#dense layers
dense2Layer = tfkl.Dense(32, activation = "relu")(outLayerConcat)
outLayer = tfkl.Dense(1, activation = "linear")(dense2Layer)
#compiling model
mcMod = tfkm.Model(inputs=[textInLayer, numInLayer], outputs=outLayer)
mcMod.compile(optimizer = "adam", loss = "mean_squared_error", metrics = ["mae"])
mcMod.summary()

trainingCycles = 3
for trainingCycle in range(trainingCycles):
    chunkLoopCount = int(np.ceil(mcEntryCount / rootChunkSize))
    for rootChunkLoopCounter in range(chunkLoopCount):
        gc.collect()
        print(f"Chunk {rootChunkLoopCounter + 1}/{chunkLoopCount} // {np.round((rootChunkLoopCounter / chunkLoopCount) * 100, 2)}% Complete // Cycle {trainingCycle + 1}/{trainingCycles}")
        if rootChunkLoopCounter != 0:
            mcDataChunk = PreprocessData(daskdf.from_pandas(pd.read_csv(mcPath, names = mcHeaders, chunksize = rootChunkSize, skiprows = rootChunkSize * rootChunkLoopCounter, low_memory = False).get_chunk(rootChunkSize), npartitions = chunkCount), uniqueLabels)
        else:
            mcDataChunk = PreprocessData(daskdf.from_pandas(pd.read_csv(mcPath, chunksize = rootChunkSize, low_memory = False).get_chunk(rootChunkSize), npartitions = chunkCount), uniqueLabels)
        mcMod = FinalPreprocessAndTrain(mcDataChunk, langPro, mcMod, mcTokenizer, maxSeqLen, mcScaler)
        mcMod.save("intermediate/MetacriticReviewModel_chunkCheckpoint.keras")
    mcMod.save("intermediate/MetacriticReviewModel_cycleCheckpoint.keras")
mcMod.save("MetacriticReviewModel.keras")
SaveModVars()

mcMSE, mcMAE, mcR2 = EvaluateMCModel(mcMod, mcScaler, mcTokenizer, maxSeqLen, uniqueLabels)
print(f"Evaluation -- MSE: {mcMSE}, MAE: {mcMAE}, R2: {mcR2}")

mcScaler, mcTokenizer, maxSeqLen, uniqueLabels = LoadModVars()
inData = {
    "Game Release Date": "27-Feb-25",
    "Game Developer": "Nintendo",
    "Genre": "Genre(s):Miscellaneous,Puzzle,Action,Platformer",
    "Platforms": "PlayStation5,Switch,PC",
    "Product Rating": "T",
    "Overall Metascore": 75,
    "Overall User Rating": "8.5",
    "Rating Given By The Reviewer": "8",
    "Review Text": "The open-world gameplay is makes this platformer highly enjoyable."
}
mcPred = PredictMetascore(mcMod, mcTokenizer, mcScaler, maxSeqLen, inData)
print(f"Predicted Metascore: {mcPred[0][0]}")