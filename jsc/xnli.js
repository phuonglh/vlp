let line = '{"annotator_labels": ["neutral", "contradiction", "neutral", "neutral", "neutral"], "genre": "facetoface", "gold_label": "neutral", "language": "vi", "match": "True", "pairID": "1", "promptID": "1", "sentence1": "V\u00e0 anh \u1ea5y n\u00f3i, M\u1eb9, con \u0111\u00e3 v\u1ec1 nh\u00e0.", "sentence1_tokenized": "V\u00e0 anh \u1ea5y n\u00f3i , M\u1eb9 , con \u0111\u00e3 v\u1ec1 nh\u00e0 .", "sentence2": "Ngay khi xu\u1ed1ng xe bu\u00fdt c\u1ee7a tr\u01b0\u1eddng, anh \u1ea5y g\u1ecdi cho m\u1eb9.", "sentence2_tokenized": "Ngay khi xu\u1ed1ng xe bu\u00fdt c\u1ee7a tr\u01b0\u1eddng , anh \u1ea5y g\u1ecdi cho m\u1eb9 ."}';
let object = JSON.parse(line)
console.log(object["sentence1"]);
console.log(object["sentence2"]);
