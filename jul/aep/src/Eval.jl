# phuonglh
# Evaluate the performance of classifier and parser, write scores
# to JSON file

using JSON3
using JSONTables
using DataFrames
using Statistics


include("Parser.jl")

"""
    experiment(options, times=3)

    Perform experimentation with a given number of times.
"""
function experiment(options, times=3)
    scorePath = options[:scorePath]
    file = if isfile(scorePath)
        open(scorePath, "a")
    else
        open(scorePath, "w")
    end

    # Vietnamese
    for t = 1:times
        local elapsedTime = time_ns()
        train(options)
        elapsedTime = time_ns() - elapsedTime
        sentencesTrain = readCorpus(options[:trainCorpus], options[:maxSequenceLength])
        sentencesDev = readCorpus(options[:devCorpus], options[:maxSequenceLength])
        sentencesTest = readCorpus(options[:testCorpus], options[:maxSequenceLength])
        accuracyTrain = eval(options, sentencesTrain)
        accuracyDev = eval(options, sentencesDev)
        accuracyTest = eval(options, sentencesTest)
        trainingUAS, trainingLAS = evaluate(options, sentencesTrain)
        devUAS, devLAS = evaluate(options, sentencesDev)
        testUAS, testLAS = evaluate(options, sentencesTest)
        local scores = Dict{Symbol,Any}(
            :trainCorpus => options[:trainCorpus],
            :minFreq => options[:minFreq],
            :maxSequenceLength => options[:maxSequenceLength],
            :wordSize => options[:wordSize],
            :shapeSize => options[:shapeSize],
            :posSize => options[:posSize], 
            :embeddingSize => options[:embeddingSize],
            :hiddenSize => options[:hiddenSize],
            :batchSize => options[:batchSize],
            :trainingTime => elapsedTime,
            :trainingAccuracy => accuracyTrain,
            :developmentAccuracy => accuracyDev,
            :testAccuracy => accuracyTest,
            :trainingUAS => trainingUAS,
            :trainingLAS => trainingLAS,
            :devUAS => devUAS,
            :devLAS => devLAS,
            :testUAS => testUAS,
            :testLAS => testLAS
        )
        line = JSON3.write(scores)
        write(file, string(line, "\n"))
        flush(file)
    end
    close(file)
end

"""
    toDF(options)

    Load experimental results into a data frame for analysis.
"""
function toDF(options)
    # read lines from the score path, concatenate them into an json array object
    lines = readlines(options[:scorePath])
    s = string("[", join(lines, ","), "]")
    # convert to a json table
    jt = jsontable(s)
    # convert to a data frame
    DataFrame(jt)
end


"""
    analyse(options)

    Analyse the experimental results.
"""
function analyse(options)
    df = toDF(options)
    # select test scores and hidden size to see the effect of varying hidden size
    testScores = select(df, [:hiddenSize, :testAccuracy, :testUAS, :testLAS])
    # group test scores by hidden size
    gdf = groupby(testScores, :hiddenSize)
    # compute mean scores for each group
    combine(gdf, names(gdf) .=> mean)
end