require('dotenv').config();
const express = require("express");
const bodyParser = require("body-parser");
const tf = require("@tensorflow/tfjs-node");

const {
  validateImageType,
  scanLimitter,
  multerMiddleware,
} = require("./utils");
// const {onRequest} = require("firebase-functions/v1/https");

const server = express();
const PORT = process.env.PORT || 8080;
const URL = process.env.MODEL_URL;

server.disable("x-powered-by");
server.use(bodyParser.json());
server.use(bodyParser.urlencoded({extended: false}));

server.get("/", (req, res) => {
  res.status(200).json({
    message: "Response successful",
    success: true,
  });
});

server.post("/scan", scanLimitter, multerMiddleware.single("file"), async (req, res, next) => {
      const file = req.file;

      if (!file) {
        return res.status(404).json({
          message: "Error, missing file",
        });
      }

      if (!validateImageType(file)) {
        return res.status(404).json({
          message: "Error, invalid file type",
        });
      }

      try {
        const {buffer} = file;
        const model = await tf.loadLayersModel(URL);

        const tensor = tf.tidy(() => {
          const decode = tf.node.decodeImage(buffer, 3);
          const resize = tf.image.resizeBilinear(decode, [224, 224]);
          const normalize = tf.div(resize.toFloat(), 255.0);
          const expand = tf.expandDims(normalize, 0);

          return expand;
        });

        const classes = model.predict(tensor);
        const flatten = classes.flatten()
        const label = [true, false];
        const tensorToArray = await flatten.array();
        const predictedIndex = tensorToArray.reduce(
            (max, x, i, arr) => x > arr[max] ? i : max, 0,
        );
        const isBatik = label[predictedIndex];

        res.status(200).json({
          isBatik,
          message: "Image has been scanned successfully",
          success: true,
        });
      } catch (error) {
        next(error);
      }
    });

server.use((err, req, res, _next) => {
  console.log(err);
  res.status(500).json({
    message: "Something went wrong",
  });
});

server.listen(PORT, () => {
  console.log(`Listening on PORT ${PORT}`);
});

