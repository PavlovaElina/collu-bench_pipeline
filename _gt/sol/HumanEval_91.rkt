#lang racket
(define (is_bored S)
  (define sentences (regexp-split #px"[.?!]" S))
  (for/sum ([sent sentences]
            #:when (let ([t (string-trim sent)])
                     (or (string=? t "I") (string-prefix? t "I ")))) 1))
