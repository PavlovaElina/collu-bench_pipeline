#lang racket
(define (generate_integers a b)
  (define lo (max 2 (min a b)))
  (define hi (min 8 (max a b)))
  (for/list ([x (in-range lo (+ hi 1))] #:when (even? x)) x))
