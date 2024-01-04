/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import { format } from "d3-format";
import performanceMetrics from "~/constants/performanceMetrics";

const EQUAL_OPPORTUNITY = "equal_opportunity";
const PREDICTIVE_EQUALITY = "predictive_equality";

const capitalize = (text) => text.replace(/\b\w/g, (l) => l.toUpperCase());

export const formatPercentage = (value) => format(".1%")(value);

export const formatNumber = (value) => format(".3")(value);

export const formatLargeInteger = (value) => {
    const formattedValue = format(".3s")(value);

    switch (formattedValue[formattedValue.length - 1]) {
        case "G":
            return `${formattedValue.slice(0, -1)}B`;
    }
    return formattedValue;
};

export const formatPercentagePoint = (value, showSign = true) => {
    const formatTemplate = `${showSign ? "+" : ""}.1%`;

    return format(formatTemplate)(value).replace("%", " pp");
};

export const formatPerformanceLabel = (metric) => {
    const metricName = metric.replace("money_", "");

    if (Object.prototype.hasOwnProperty.call(performanceMetrics, metricName)) {
        const prefix = metric.includes("money_") ? "$" : "#";

        return `${prefix} ${performanceMetrics[metricName]}`;
    }

    return metric;
};

export const formatFairnessLabel = (fairnessMetric) => {
    const aggregatorLabels = {
        "min": "minimum",
        "max": "maximum",
    };

    if (fairnessMetric.includes("disparity")) {
        /* The fairness metrics on disparity follow the structure 
        <protected_col>_<aggregator>_<metric>_disparity. In order to format this metric 
        to be human-readable, we use regex expressions to capture the different parts of
        the string  */

        /* The performance metric is the last metric before the '_disparity' keyword */
        const perfRegex = new RegExp("(?<=_)[a-z]*(?=_disparity)", "g");
        const perfMetric = fairnessMetric.match(perfRegex)[0];
        const perfMetricLabel = formatPerformanceLabel(perfMetric);

        /* The aggregator, that can be "min" or "max" is the metric that precedes the 
        performance metric */
        const aggregatorRegex = new RegExp(`(?<=_)[a-z]*(?=_${perfMetric})`, "g");
        const aggregator = fairnessMetric.match(aggregatorRegex)[0];

        /* The protected column, is the first segment of the string before the first _ */
        const protectedColumnRegex = new RegExp(`^.*(?=_${aggregator})`, "g");
        const protectedColumn = fairnessMetric.match(protectedColumnRegex)[0];

        return `[${capitalize(protectedColumn)}] ${capitalize(
            aggregatorLabels[aggregator],
        )} Disparity on ${perfMetricLabel}`;
    }

    if (fairnessMetric.includes(EQUAL_OPPORTUNITY) || fairnessMetric.includes(PREDICTIVE_EQUALITY)) {
        const fairMetricRegex = new RegExp(`(${EQUAL_OPPORTUNITY}|${PREDICTIVE_EQUALITY})`);
        let fairMetric = fairnessMetric.match(fairMetricRegex)[0].replace("_", " ");

        if (fairnessMetric !== EQUAL_OPPORTUNITY && fairnessMetric !== PREDICTIVE_EQUALITY) {
            const protectedColumnRegex = new RegExp(`^.*(?=_(${EQUAL_OPPORTUNITY}|${PREDICTIVE_EQUALITY}))`);
            const protectedColumn = fairnessMetric.match(protectedColumnRegex)[0];

            fairMetric = `[${protectedColumn}] ${fairMetric}`;
        }
        return capitalize(fairMetric);
    }
    return fairnessMetric;
};
