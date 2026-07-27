#lang racket
(define (all_prefixes s)
  (for/list ([i (in-range 1 (+ (string-length s) 1))])
    (substring s 0 i)))
