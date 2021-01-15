var path = require("path");

const config = {
  entry: ["./js/index.js"],
  output: {
    path: __dirname + "/dist",
    filename: "aequitas.js",
    library: "aequitas",
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        loader: "babel-loader",
        options: {
          presets: ["@babel/preset-env", "@babel/react"],
        },
      },
      {
        test: /\.s[ac]ss$/i,
        use: ["style-loader", "css-loader", "sass-loader"],
      },
      {
        test: /\.(png|svg|jpg|gif)$/,
        use: [
          {
            loader: "file-loader",
            options: {
              name: "images/[hash]-[name].[ext]",
            },
          },
        ],
      },
    ],
  },
  resolve: {
    extensions: [".js"],
    alias: {
      ["~"]: path.resolve(__dirname, "js"),
    },
  },
  devServer: {
    writeToDisk: true,
    hot: true,
    inline: false,
  },
  mode: "development",
};
module.exports = config;
