import dotenv from "dotenv";
dotenv.config();
// Import required modules
import express from "express";
import pkg from "body-parser";
const { json } = pkg;
import cors from "cors";
import axios from "axios";
import { tidy, dot, mul } from "@tensorflow/tfjs";
import { load } from "@tensorflow-models/universal-sentence-encoder";
import { contextRows } from "./constants.js";

// Define the environment variables and initialize the express application
const PORT = process.env.PORT;
const apiKey = process.env.API_KEY;
const apiUrl = process.env.API_URL;
const app = express();

app.use(cors());
app.use(json());

let model;
let contextEmbeddings;

// Define API URL and headers for external API requests
const API_URL = apiUrl;
const headers = {
  Accept: "application/json",
  Authorization: apiKey,
  "Content-Type": "application/json",
};

// Predefined contexts for embedding and similarity comparison

// Load the TensorFlow model and compute embeddings for predefined contexts
async function loadModelAndEmbedContext() {
  model = await load();
  contextEmbeddings = await model.embed(contextRows);
}

loadModelAndEmbedContext().then(() => {
  console.log("Model loaded and context embeddings precomputed");
});

// Calculate cosine similarity between two vectors
function cosineSimilarity(vecA, vecB) {
  const dotProduct = tidy(() => {
    const a = vecA.flatten();
    const b = vecB.flatten();
    return dot(a, b);
  });
  const magnitudeA = vecA.norm();
  const magnitudeB = vecB.norm();
  const cosineSimilarity = dotProduct.div(mul(magnitudeA, magnitudeB));
  return cosineSimilarity.dataSync()[0];
}

// Find the most similar context index based on user query
async function findMostSimilarIndex(userQuery) {
  const queryEmbedding = await model.embed([userQuery]);
  let highestSimilarity = -1;
  let mostSimilarIndex = -1;

  for (let i = 0; i < contextRows.length; i++) {
    const summaryEmbedding = contextEmbeddings.slice([i, 0], [1]);
    const similarity = cosineSimilarity(
      queryEmbedding.squeeze(),
      summaryEmbedding.squeeze()
    );
    if (similarity > highestSimilarity) {
      highestSimilarity = similarity;
      mostSimilarIndex = i;
    }
  }

  return mostSimilarIndex;
}

// Function to query an external API with the context and question
async function queryExternalAPI(context, question) {
  try {
    const payload = {
      inputs: {
        question: question,
        context: context,
      },
    };
    const response = await axios.post(API_URL, payload, { headers });
    return response.data;
  } catch (error) {
    console.error("Error querying external API:", error);
    throw error; // Rethrow error to handle it in the calling function
  }
}

// Endpoint to handle queries from the frontend
app.post("/query", async (req, res) => {
  const userQuery = req.body.userString;
  try {
    // Find the index of the most similar summary
    const mostSimilarIndex = await findMostSimilarIndex(userQuery);

    // Ensure a valid index is found; otherwise, send an error response
    if (mostSimilarIndex === -1) {
      return res
        .status(400)
        .send("No suitable context found for the provided query.");
    }

    // Use the most similar context for the question
    const mostRelevantContext = contextRows[mostSimilarIndex];

    // Query the external API with the selected context and the user's question
    const externalAPIResponse = await queryExternalAPI(
      mostRelevantContext,
      userQuery
    );
    res.json(externalAPIResponse); // Send the external API's response back to the client
  } catch (error) {
    console.error("Error while querying the external API:", error);
    res.status(500).send("Failed to get a response from the external API.");
  }
});

// Basic endpoint to check server response
app.get("/", (req, res) => {
  res.json({ message: "Hello from server!" });
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server listening on ${PORT}`);
});
