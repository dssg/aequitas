/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2022 Feedzai, Strictly Confidential
 */

import chartSettings from "~/constants/scatterplot";


 export const translation = (newDomain, prevTranslate, newTranslate, currScale, limitMin, limitMax, scale, initValue) => {
    if (newDomain[0] < chartSettings.minDomain && newDomain[1] > chartSettings.maxDomain) {
        return prevTranslate;
    }

    else if (newDomain[0] < chartSettings.minDomain) {
        return scale(initValue + (initValue - limitMin) * currScale);
    }
    
    else if (newDomain[1] > chartSettings.maxDomain) {
        return scale(initValue + (initValue - limitMax) * currScale);
    }

    return newTranslate;
}

export const calculateDomain = (scale, applyTranslation, applyScale) => {
    return scale
        .range()
        .map((r) =>
            scale.invert(
            (r - applyTranslation) / applyScale
            )
        );
}

export const adjustDomain = (newDomain, prevDomain, newScale, prevScale) => {
    var adjustedDomain = [Math.max(newDomain[0], chartSettings.minDomain), Math.min(newDomain[1], chartSettings.maxDomain)];

    // prevent zoom issue
    // when the upper limit of X domain is 1 or the lower limit of the Y domain is 0
    // the zoom out action would make those values change when they shouldn't
    if (newScale < prevScale) {
        if (newDomain[1] < prevDomain[1]) {
            adjustedDomain = [adjustedDomain[0], chartSettings.maxDomain];
        }
        
        if (newDomain[0] > prevDomain[0]) {
            adjustedDomain = [chartSettings.minDomain, adjustedDomain[1]];
        }
    }
    
    return adjustedDomain;
}

export const rescaleAxis = (scale, applyTranslation, applyScale) => {
    var newDomain = calculateDomain(scale, applyTranslation, applyScale);

    return scale.copy().domain(
        [Math.max(newDomain[0], chartSettings.minDomain), Math.min(newDomain[1], chartSettings.maxDomain)]);
};