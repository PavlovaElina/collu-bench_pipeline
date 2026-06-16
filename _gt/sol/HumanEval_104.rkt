#lang racket
(define (unique_digits x)
  (define (no-even? n)
    (for/and ([c (in-string (number->string n))]) (odd? (- (char->integer c) 48))))
  (sort (filter no-even? x) <))
