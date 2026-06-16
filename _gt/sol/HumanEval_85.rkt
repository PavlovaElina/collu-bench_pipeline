#lang racket
(define (add lst)
  (for/sum ([x (in-list lst)] [i (in-naturals)]
            #:when (and (odd? i) (even? x))) x))
