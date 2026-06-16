#lang racket
(define (count_upper s)
  (for/sum ([c (in-string s)] [i (in-naturals)]
            #:when (and (even? i) (member c '(#\A #\E #\I #\O #\U)))) 1))
