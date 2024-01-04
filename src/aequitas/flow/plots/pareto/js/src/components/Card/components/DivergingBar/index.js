/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import React, { useMemo } from "react";
import PropTypes from "prop-types";
import { Line } from "@visx/shape";
import { scaleLinear } from "@visx/scale";

import chartSettings from "./chartConfig";
import { useChartDimensions } from "~/utils/hooks";
import { formatPercentagePoint } from "~/utils/formatters";
import { useAppState } from "~/utils/context";
import labels from "~/enums/labels";

export default function DivergingBar({ metric }) {
    const { pinnedModel, selectedModel } = useAppState();
    const [wrapperDivRef, dimensions] = useChartDimensions(chartSettings.dimensions);

    const xScale = useMemo(
        () =>
            scaleLinear({
                domain: [-1, 1],
                range: [0, dimensions.width],
            }),
        [dimensions.width],
    );

    const difference = selectedModel[metric] - pinnedModel[metric];
    const diffPosition = xScale(difference);
    const barLength = Math.abs(diffPosition - xScale(0));
    const barHeight = dimensions.height * 0.8;

    const barX = difference >= 0 ? xScale(0) : diffPosition;
    const color = difference >= 0 ? "green" : "orange";

    const isDifferenceZero = Math.abs(difference) < 0.001;
    const wrapperClass = `chart-wrapper ${!isDifferenceZero ? color : null}`;

    return (
        <td className="value chart-cell">
            <div ref={wrapperDivRef} className={wrapperClass} data-testid="diverging-rect">
                <svg width={dimensions.width} height={dimensions.height}>
                    <Line
                        from={{ x: dimensions.width / 2, y: 0 }}
                        to={{ x: dimensions.width / 2, y: dimensions.height }}
                        className="reference-line"
                    />
                    <rect
                        x={barX}
                        y={(dimensions.height - barHeight) / 2}
                        width={barLength}
                        height={barHeight}
                        className={color}
                    />
                </svg>
                <span>{!isDifferenceZero ? formatPercentagePoint(difference) : labels.EQUAL}</span>
            </div>
        </td>
    );
}

DivergingBar.propTypes = {
    metric: PropTypes.string.isRequired,
};
