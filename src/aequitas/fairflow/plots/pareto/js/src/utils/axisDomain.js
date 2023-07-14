/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2022 Feedzai, Strictly Confidential
 */

import chartSettings from "~/constants/scatterplot";

export const domainSize = (interval) => {
    /**
    multiply by 100 to fix the floating point number precision issue
    Source: https://www.avioconsulting.com/blog/overcoming-javascript-numeric-precision-issues#:~:text=In%20Javascript%2C%20all%20numbers%20are,the%20sign%20in%20bit%2063.
    **/

    return (interval[1]*100 - interval[0]*100)/100;
}

const roundToFiveMultiple = (val, index) => index === 0 ? Math.floor((val*100)/5)/100*5 : Math.ceil((val*100)/5)/100*5;

export const roundDomain = (domain) => {
    if (domainSize(domain) <= chartSettings.minDistance)
        return domain;
    
    return domain.map((val, index) => roundToFiveMultiple(val, index));
}