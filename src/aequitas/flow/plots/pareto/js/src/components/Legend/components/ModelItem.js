/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import React from "react";
import PropTypes from "prop-types";
import { GlyphCircle, GlyphStar } from "@visx/glyph";

import chartSettings from "~/constants/scatterplot";

function ModelItem({ pointClassName, labelText, isRecommended }) {
    const Point = isRecommended ? GlyphStar : GlyphCircle;

    return (
        <div className="legend-item">
            {labelText}

            <svg
                width={chartSettings.pointSize / 2}
                height={chartSettings.pointSize / 2}
                className="legend-item-symbol"
            >
                <Point
                    top={chartSettings.pointSize / 4}
                    left={chartSettings.pointSize / 4}
                    size={chartSettings.pointSize}
                    className={pointClassName}
                />
            </svg>
        </div>
    );
}

ModelItem.propTypes = {
    pointClassName: PropTypes.string.isRequired,
    labelText: PropTypes.string.isRequired,
    isRecommended: PropTypes.bool,
};

ModelItem.defaultProps = {
    isRecommended: false,
};

export default ModelItem;
