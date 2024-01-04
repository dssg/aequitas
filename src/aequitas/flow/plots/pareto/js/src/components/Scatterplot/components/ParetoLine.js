/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import React from "react";
import PropTypes from "prop-types";
import { LinePath } from "@visx/shape";

import { useTunerState } from "~/utils/context";

function ParetoLine({ paretoPoints, xScale, yScale }) {
    const { optimizedFairnessMetric, optimizedPerformanceMetric } = useTunerState();
    const sortedParetoPoints = paretoPoints
        .sort((a, b) => b[optimizedPerformanceMetric] - a[optimizedPerformanceMetric])
        .sort((a, b) => a[optimizedFairnessMetric] - b[optimizedFairnessMetric]);

    return (
        <LinePath
            data={sortedParetoPoints}
            x={(d) => xScale(d[optimizedPerformanceMetric])}
            y={(d) => yScale(d[optimizedFairnessMetric])}
            id="line"
            className="pareto-line"
            data-testid="pareto-line"
        />
    );
}

ParetoLine.propTypes = {
    paretoPoints: PropTypes.arrayOf(PropTypes.object).isRequired,
    xScale: PropTypes.func.isRequired,
    yScale: PropTypes.func.isRequired,
};

export default ParetoLine;
