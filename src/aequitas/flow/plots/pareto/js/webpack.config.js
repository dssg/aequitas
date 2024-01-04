/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

const path = require("path");
const autoprefixer = require("autoprefixer");

const config = {
    entry: ["./src/index.js"],
    output: {
        path: __dirname + "/dist",
        filename: "fairAutoML.js",
        library: "fairAutoML",
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
                test: /\.(scss|css)$/,
                use: [
                    "style-loader",
                    "css-loader",
                    "sass-loader",
                    {
                        loader: "postcss-loader",
                        options: {
                            plugins: () => [autoprefixer()],
                        },
                    },
                ],
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
            {
                test: /\.(woff|woff2|eot|ttf|otf)$/,
                type: "asset/inline",
            },
        ],
    },
    resolve: {
        alias: {
            "~": path.resolve(__dirname, "src"),
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
