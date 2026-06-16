#lang racket
(define (solve N)
  (define s (for/sum ([c (in-string (number->string N))]) (- (char->integer c) 48)))
  (number->string s 2))
