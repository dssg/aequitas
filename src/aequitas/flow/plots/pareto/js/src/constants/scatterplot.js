/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import { scatterplotSizePx } from "./_sizes.scss";

const scatterplotSize = parseInt(scatterplotSizePx.replace("px", ""));

export default {
    dimensions: {
        marginLeft: 40,
        marginRight: 10,
        marginTop: 10,
        marginBottom: 50,
        width: scatterplotSize,
        height: scatterplotSize,
    },
    numTicks: 4,
    pointSize: 40,
    minDistance: 0.02,
    minDomain: 0,
    maxDomain: 1,
    initialTransform: {
        scaleX: 1,
        scaleY: 1,
        translateX: 0,
        translateY: 0,
        skewX: 0,
        skewY: 0
    },
};