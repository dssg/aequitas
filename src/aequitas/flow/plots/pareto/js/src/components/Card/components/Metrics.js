/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import React from "react";
import PropTypes from "prop-types";
import { isNull } from "lodash";

import DivergingBar from "./DivergingBar";
import FairnessExplainer from "./FairnessExplainer";
import { useAppState } from "~/utils/context";
import { formatPercentage } from "~/utils/formatters";

function Metrics({ metrics, optimizedMetric, title, metricLabelFormatter, isFairness }) {
    const { pinnedModel, selectedModel } = useAppState();
    const showComparison = !isNull(selectedModel);

    const models = [pinnedModel, selectedModel].filter(Boolean);
    const sortedMetrics = metrics.sort((a, b) => {
        if (a === optimizedMetric) return -1;
        if (b === optimizedMetric) return 1;
        return 0;
    });

    return (
        <>
            <tr className="title">
                <td>
                    {title} {isFairness ? <FairnessExplainer /> : null}
                </td>
            </tr>
            {sortedMetrics.map((metric) => {
                const isOptimized = metric === optimizedMetric;
                const metricClass = isOptimized ? "bold" : "";

                return (
                    <tr className={`metrics-row ${metricClass}`} key={`table-${title}-${metric}`}>
                        <td className={`metrics-row-title`}>{metricLabelFormatter(metric)}</td>
                        {models.map((model, index) => {
                            const isPinned = index === 0;

                            return (
                                <td className="value" key={`col-${isPinned}-${metric}`}>
                                    {formatPercentage(model[metric])}
                                </td>
                            );
                        })}
                        {showComparison ? <DivergingBar metric={metric} /> : null}
                    </tr>
                );
            })}
        </>
    );
}

Metrics.propTypes = {
    metrics: PropTypes.arrayOf(PropTypes.string).isRequired,
    optimizedMetric: PropTypes.string.isRequired,
    title: PropTypes.string.isRequired,
    metricLabelFormatter: PropTypes.func.isRequired,
    isFairness: PropTypes.bool,
};

Metrics.defaultPropTypes = {
    isFairness: false,
};

export default Metrics;
