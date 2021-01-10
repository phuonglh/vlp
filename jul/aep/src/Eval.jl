# phuonglh
# Evaluate the performance of classifier and parser, write scores
# to JSON file

using JSON3

include("Parser.jl")

times = 1

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
end
close(file)
