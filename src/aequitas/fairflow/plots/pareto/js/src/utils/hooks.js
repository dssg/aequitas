/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import { useRef } from "react";

/* useChartDimensions Hook
Hook to help with scaling the chart to be responsive. 
From https://wattenberger.com/blog/react-hooks 
*/
const combineChartDimensions = (dimensions) => {
    const parsedDimensions = {
        marginTop: 40,
        marginRight: 30,
        marginBottom: 40,
        marginLeft: 75,
        ...dimensions,
    };

    return {
        ...parsedDimensions,
        boundedHeight: Math.max(
            parsedDimensions.height - parsedDimensions.marginTop - parsedDimensions.marginBottom,
            0,
        ),
        boundedWidth: Math.max(parsedDimensions.width - parsedDimensions.marginLeft - parsedDimensions.marginRight, 0),
    };
};

export const useChartDimensions = (passedSettings) => {
    const ref = useRef();
    const dimensions = combineChartDimensions(passedSettings);

    return [ref, dimensions];
};
