#lang racket
(define (solution lst)
  (for/sum ([x (in-list lst)] [i (in-naturals)]
            #:when (and (even? i) (odd? x))) x))
