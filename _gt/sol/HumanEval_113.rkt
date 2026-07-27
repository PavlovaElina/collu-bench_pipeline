#lang racket
(define (odd_count lst)
  (map (lambda (s)
         (define cnt (for/sum ([ch (in-string s)] #:when (odd? (- (char->integer ch) 48))) 1))
         (define c (number->string cnt))
         (string-append "the number of odd elements " c "n the str" c "ng " c " of the " c "nput."))
       lst))
